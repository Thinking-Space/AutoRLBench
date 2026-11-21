"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import logging
import os
import random
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
        idempotent: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


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
    template_env = ("SLIME_AGENT_SANDBOX_TEMPLATE", "E2B_TEMPLATE")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")

    default_lifetime_sec = 3600
    default_rpc_retries = 3
    cleanup_rpc_attempts = 2
    # With retries=3 the sleep budget is roughly 3s plus jitter, which handles
    # common SSL / protocol / pool-timeout flaps during safe-to-retry operations.
    rpc_backoff_base_sec = 1.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        template: str | None = None,
        rpc_retries: int | None = None,
        strict_read: bool = False,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.template = template or _getenv(*self.template_env) or None
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.strict_read = strict_read
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

    @staticmethod
    def _is_transient_rpc_error(e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in {
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
            "ClosedResourceError",
            "BrokenResourceError",
        }:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(
        self,
        op_name: str,
        coro_factory,
        *,
        attempts: int | None = None,
    ):
        """Run coro_factory() with retries for transient E2B RPC failures."""
        attempts = self.rpc_retries if attempts is None else attempts
        if attempts < 1:
            raise ValueError(f"retry attempts must be positive, got {attempts}")
        last_err = None
        for attempt in range(attempts):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                last_err = e
                if attempt + 1 < attempts:
                    backoff = self.rpc_backoff_base_sec * (2**attempt) * random.uniform(0.5, 1.5)
                    logger.warning(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        attempts,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = {self.image_metadata_key: self.image}
        create_kwargs = {}
        if api_key := os.environ.get("E2B_API_KEY"):
            create_kwargs["api_key"] = api_key
            domain = os.environ.get("E2B_DOMAIN", "")
            if domain and domain != "e2b.app" and not api_key.startswith("e2b_"):
                try:
                    from e2b.connection_config import ConnectionConfig  # type: ignore

                    supported = "validate_api_key" in inspect.signature(ConnectionConfig).parameters
                except (ImportError, TypeError, ValueError):
                    supported = False
                if supported:
                    create_kwargs["validate_api_key"] = False
        if proxy := os.environ.get("http_proxy"):
            create_kwargs["proxy"] = proxy
        self._sb = await self._rpc_retry(
            "create",
            lambda: AsyncSandbox.create(
                template=self.template,
                timeout=self.timeout,
                metadata=md,
                **create_kwargs,
            ),
            attempts=self.rpc_retries,
        )
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._rpc_retry("kill", self._sb.kill, attempts=self.cleanup_rpc_attempts)
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
        idempotent: bool = False,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

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
                attempts=self.rpc_retries if idempotent else 1,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
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

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            if self.strict_read:
                raise
            return ""


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /workspace {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
        idempotent=True,
    )
