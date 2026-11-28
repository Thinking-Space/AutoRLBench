"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    ``idempotent`` is a hint for the backend's transport-retry policy: callers
    mark whether re-sending the command after a severed response is safe to
    replay (see ``E2BSandbox._rpc_retry``). Backends without retries may
    ignore it.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


def _effective_user(user: str) -> str:
    """Remap the sandbox-side user when the gateway requires it.

    The AGS-compatible gateway's envd cannot run a non-root user through a
    workspace rooted under /root (su works, envd exec does not — each exec
    appears isolated at the namespace/MAC layer); production rollouts for
    these task families ran everything as root. E2B_FORCE_USER remaps every
    exec/files-API user, which also matches that setup."""
    override = os.environ.get("E2B_FORCE_USER", "").strip()
    return override or user


_PROXY_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


def _strip_proxy_env() -> dict[str, str]:
    """Remove proxy env vars so httpx clients created in this scope use no
    proxy; returns what was stripped for restore."""
    stripped = {}
    for var in _PROXY_ENV_VARS:
        if var in os.environ:
            stripped[var] = os.environ.pop(var)
    return stripped


def _restore_proxy_env(stripped: dict[str, str]) -> None:
    for k, v in stripped.items():
        os.environ[k] = v


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    The 5s ``test -f && cat`` polls are deliberately short, idempotent RPCs --
    they keep the sandbox alive against idle GC while the detached command runs
    over a stream the gateway can't sever.
    """
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done_file} && cat {done_file}", user=user, timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    and the polling doubles as an idle-GC keepalive while the command runs.
    """
    # Scope the marker/launcher paths to the calling sandbox: concurrent
    # episodes share /tmp on local backends and must not collide.
    sbx = (getattr(sb, "sandbox_id", "") or "shared")[:8]
    tag = f"{tag}.{sbx}"
    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    # Sandboxes with a private home dir (local backend) get it honored here.
    home = getattr(sb, "home", None)
    home_line = f"export HOME={home}\n" if home else f"export HOME=/home/{user}\n"
    prefix = f"cd {workdir}\n{home_line}" if workdir else ""
    launcher_body = f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    # Clear the previous invocation's state in its own idempotent RPC, *before*
    # the guarded spawn. The mkdir guard below exists only to dedupe transport
    # retries of this one spawn (a severed response replayed by _rpc_retry); it
    # must not survive into the next logical invocation of the same tag (e.g.
    # install_npm_cli's retry loop), which would skip the spawn entirely and
    # read the previous run's stale exit-code marker. Callers must not overlap
    # two exec_and_wait calls with the same tag.
    await sb.exec(
        f"rm -rf {lock_dir}; rm -f {out_file} {done_file}",
        user=user,
        timeout=30,
        check=True,
        idempotent=True,
    )
    # Local-backend sandboxes keep their commands inside the sandbox's own
    # process group (no inner setsid) so the watchdog can reap them at TTL.
    detach = "" if getattr(sb, "no_setsid", False) else "setsid "
    await sb.exec(
        f"chmod +x {launcher}; "
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"{detach}bash {launcher} < /dev/null > {out_file} 2>&1 &",
        user=user,
        env=env,
        timeout=30,
        check=True,
        idempotent=True,
    )
    exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 512 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("SLIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
        ep_dir: str | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.size = size if size is not None else self._size_from_env()
        # per-episode archive dir (host side): backend event journal + death
        # telemetry for the rollout loop
        self.ep_dir = ep_dir
        self._ttl_sec = None
        self._died_at = None
        self._ttl_task = None
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @classmethod
    def _size_from_env(cls) -> str:
        return _getenv(*cls.size_env, default=cls.default_size)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._reset_conn_pool()
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        from e2b import AsyncSandbox  # type: ignore

        if os.environ.get("TENCENT_AGS_DOMAIN", "").strip():
            # AGS-compatible gateway: connection settings come from
            # E2B_API_KEY / E2B_API_URL / E2B_DOMAIN; instances are routed by
            # the full x-custom-config payload (image via the enterprise
            # registry, envd entrypoint, resources) plus the gateway template.
            cfg = {
                "image": self._ags_rewrite_image(self.image),
                "imageRegistryType": os.environ.get("TENCENT_AGS_IMAGE_REGISTRY_TYPE", "enterprise"),
                "command": ["/bin/sh", "-c"],
                "args": ["/mnt/envd -port 49983"],
                "resources": {
                    "cpu": os.environ.get("TENCENT_AGS_SANDBOX_CPU", "2"),
                    "memory": os.environ.get("TENCENT_AGS_SANDBOX_MEM", "4Gi"),
                },
            }
            md = {"x-custom-config": json.dumps(cfg)}
            ags_proxy = os.environ.get("E2B_PROXY") or None
            if ags_proxy is None:
                # Without an explicit E2B_PROXY the SDK's httpx client reads
                # http_proxy/https_proxy from the environment; on clusters
                # where the AGS endpoint is directly reachable (and the proxy
                # actively breaks the TLS handshake), strip those vars around
                # the client creation.
                _stripped = _strip_proxy_env()
            else:
                _stripped = None
            try:
                self._sb = await self._rpc_retry(
                    "create",
                    lambda: AsyncSandbox.create(
                        timeout=self.timeout,
                        metadata=md,
                        template=os.environ.get("TENCENT_AGS_TEMPLATE", "main"),
                        proxy=ags_proxy,
                    ),
                )
            finally:
                if _stripped is not None:
                    _restore_proxy_env(_stripped)
            self.sandbox_id = self._sb.sandbox_id
            # Persistent instances (the openclaw template family) ignore both
            # the create timeout and set_timeout; the instance-lifetime cap
            # is enforced host-side: a watchdog kills the instance at TTL,
            # after which every RPC fails exactly like a recycled instance.
            raw_ttl = os.environ.get("SLIME_AGENT_SANDBOX_TTL_SEC", "").strip()
            self._ttl_sec = int(raw_ttl) if raw_ttl else None
            self._log_event("created", ttl_sec=self._ttl_sec, template=os.environ.get("TENCENT_AGS_TEMPLATE"))
            if self._ttl_sec:
                self._ttl_task = asyncio.create_task(self._ags_ttl_watchdog(self._ttl_sec))
            return self

        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )

        md = {self.image_metadata_key: self.image}

        if self.size:
            prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
            size_key = f"{prefix}/size" if prefix else "size"
            md[size_key] = self.size

        self._sb = await self._rpc_retry("create", lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md))
        self.sandbox_id = self._sb.sandbox_id
        return self

    @staticmethod
    def _ags_rewrite_image(image: str) -> str:
        """Apply TENCENT_AGS_IMAGE_REWRITE prefix rules (from=to pairs)."""
        raw = os.environ.get("TENCENT_AGS_IMAGE_REWRITE", "").strip()
        for rule in raw.split(","):
            rule = rule.strip()
            if not rule or "=" not in rule:
                continue
            src, dst = rule.split("=", 1)
            if image.startswith(src):
                return dst + image[len(src):]
        return image

    async def _ags_ttl_watchdog(self, ttl_sec: int) -> None:
        await asyncio.sleep(ttl_sec)
        self._log_event("ttl_expired", ttl_sec=ttl_sec)
        try:
            await self._sb.kill()
        except Exception as e:
            self._log_event("ttl_kill_error", error=f"{type(e).__name__}: {e}")
            return
        self._died_at = time.time()
        self._log_event("killed_by_ttl")

    def _log_event(self, event: str, **fields) -> None:
        if not self.ep_dir:
            return
        try:
            import json as _json

            os.makedirs(self.ep_dir, exist_ok=True)
            rec = {"ts": round(time.time(), 3), "sandbox_id": self.sandbox_id, "event": event}
            rec.update(fields)
            with open(os.path.join(self.ep_dir, "sandbox.log"), "a") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ttl_task is not None:
            self._ttl_task.cancel()
            try:
                await self._ttl_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        user = _effective_user(user)
        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
                idempotent=idempotent,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    # The files API rejects single uploads past ~32 MiB; larger payloads go
    # through as numbered parts that the sandbox reassembles with cat.
    _WRITE_FILE_CHUNK = 24 * 1024 * 1024

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        user = _effective_user(user)
        size = self._content_size(content)
        if size is not None and size > self._WRITE_FILE_CHUNK:
            await self._write_file_chunked(sandbox_path, content, size, user=user)
            return
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    @staticmethod
    def _content_size(content: FileContent) -> int | None:
        try:
            if isinstance(content, Path):
                return content.stat().st_size
            if isinstance(content, bytes):
                return len(content)
        except OSError:
            return None
        return None

    async def _write_file_chunked(self, sandbox_path: str, content: FileContent, size: int, *, user: str) -> None:
        parts_dir = f"{sandbox_path}.parts"
        await self.exec(f"mkdir -p {parts_dir}", user=user, timeout=60, check=True)
        part_names = []
        try:
            chunk = self._WRITE_FILE_CHUNK
            if isinstance(content, Path):
                handle = open(content, "rb")
            else:
                handle = io.BytesIO(content)
            with handle:
                index = 0
                while True:
                    blob = handle.read(chunk)
                    if not blob:
                        break
                    part = f"{parts_dir}/{index:05d}"
                    part_names.append(part)

                    async def _do_part(blob=blob, part=part):
                        await self._sb.files.write(
                            part,
                            io.BytesIO(blob),
                            user=user,
                            gzip=False,
                            use_octet_stream=True,
                            request_timeout=600,
                        )

                    await self._rpc_retry(f"write_file(part {part})", _do_part)
                    index += 1
            quoted = " ".join(part_names)
            ec, _, err = await self.exec(
                f"cat {quoted} > {sandbox_path} && rm -rf {parts_dir}",
                user=user,
                timeout=300,
                check=False,
            )
            if ec != 0:
                raise RuntimeError(f"chunked write reassembly failed (exit={ec}): {err[:200]}")
        finally:
            # parts already removed on success; clear leftovers on failure
            await self.exec(f"rm -rf {parts_dir}", user=user, timeout=60, check=False)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        user = _effective_user(user)
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"(git config --system --get-all safe.directory 2>/dev/null | grep -qxF '*' || "
        f"git config --system --add safe.directory '*') && id agent",
        user="root",
        check=True,
        timeout=60,
    )
