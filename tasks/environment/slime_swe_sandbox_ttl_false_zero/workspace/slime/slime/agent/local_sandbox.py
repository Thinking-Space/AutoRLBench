"""Local-host sandbox backend for agent rollouts.

A "sandbox" here is a private directory tree under SLIME_LOCAL_SANDBOX_ROOT
plus a registry of every process this backend spawned for the episode. This
lets the rest of the agent stack (harness, task layer, evaluator) run unchanged
against the local host instead of a remote sandbox provider.

Instance recycling: ops caps the lifetime of every sandbox instance to control
host cost (see workspace notes.md, "sandbox cost optimization"). The cap is
configured through SLIME_AGENT_SANDBOX_TTL_SEC; a watchdog fires when it
expires and SIGKILLs the whole process tree. From then on the sandbox is dead
the same way a recycled cloud instance is: every later call fails, and the
caller sees an unreachable-instance error.

Every backend event (create / spawn / ttl expiry / kill / refused call) is
appended to the episode's sandbox.log so per-episode behavior can be audited
after the fact.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from slime.agent.sandbox import ExecResult, FileContent

TTL_ENV = "SLIME_AGENT_SANDBOX_TTL_SEC"
ROOT_ENV = "SLIME_LOCAL_SANDBOX_ROOT"
INSTANCE_STORE_ENV = "SLIME_INSTANCE_STORE"
DEFAULT_ROOT = "/tmp/slime-local-sbx"


class SandboxUnavailable(RuntimeError):
    """Raised when calling into a sandbox whose instance was recycled."""


class LocalSandbox:
    """Sandbox Protocol implementation over local processes and directories."""

    def __init__(self, image: str, *, ep_dir: str | None = None) -> None:
        self.image = image
        self.ep_dir = ep_dir
        self.sandbox_id = uuid.uuid4().hex[:12]
        self.root = os.path.join(os.environ.get(ROOT_ENV, DEFAULT_ROOT), self.sandbox_id)
        self.home = os.path.join(self.root, "home")
        # exec_and_wait consults this: local commands stay in the sandbox's
        # own process group so the watchdog can reap them all at once.
        self.no_setsid = True
        self._procs: dict[int, str] = {}
        self._dead = False
        self._ttl_sec: int | None = None
        self._watchdog: asyncio.Task | None = None
        self._created_at = 0.0
        self._died_at: float | None = None

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> "LocalSandbox":
        raw = os.environ.get(TTL_ENV, "").strip()
        self._ttl_sec = int(raw) if raw else None
        os.makedirs(self.home, exist_ok=True)
        self._created_at = time.time()
        self._log_event(
            "created",
            ttl_sec=self._ttl_sec,
            root=self.root,
            image=self.image,
        )
        if self._ttl_sec:
            self._watchdog = asyncio.create_task(self._ttl_watchdog(self._ttl_sec))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except (asyncio.CancelledError, Exception):
                pass
        if not self._dead:
            self._kill_all(reason="closed")

    async def _ttl_watchdog(self, ttl_sec: int) -> None:
        await asyncio.sleep(ttl_sec)
        if self._dead:
            return
        self._log_event("ttl_expired", ttl_sec=ttl_sec, uptime_sec=round(time.time() - self._created_at, 1))
        self._kill_all(reason="ttl")

    def _kill_all(self, *, reason: str) -> None:
        pids = []
        for pid in list(self._procs):
            try:
                os.kill(-pid, signal.SIGKILL)  # pid is a session/group leader
                pids.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                self._log_event("kill_permission_error", pid=pid)
        self._dead = reason == "ttl" or self._dead
        if reason == "ttl":
            self._died_at = time.time()
            self._log_event("killed_by_ttl", pids=pids)
        else:
            self._log_event("closed", pids=pids)

    # -- Sandbox protocol ----------------------------------------------------

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
        if self._dead:
            self._log_event("call_refused_after_ttl", op="exec", cmd=cmd[:80])
            raise SandboxUnavailable(
                f"sandbox {self.sandbox_id} instance was recycled (TTL) and is not reachable"
            )
        merged = self._base_env()
        if env:
            merged.update(env)
        self._log_event("exec_spawn", pid_pending=True, cmd=cmd[:120], timeout=timeout)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged,
            cwd=self.root,
            start_new_session=True,  # own group: watchdog can kill(-pid)
        )
        self._procs[proc.pid] = cmd[:120]
        try:
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                rc = proc.returncode
            except asyncio.TimeoutError:
                self._kill_pid(proc.pid)
                out, err = await proc.communicate()
                rc = 124
                self._log_event("exec_timeout", pid=proc.pid, timeout=timeout)
        finally:
            self._procs.pop(proc.pid, None)
        out_s = (out or b"").decode("utf-8", "replace")
        err_s = (err or b"").decode("utf-8", "replace")
        if check and rc != 0:
            raise RuntimeError(f"local exec failed (exit={rc}): {cmd[:120]}\n{err_s[:400]}")
        return rc, out_s, err_s

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if self._dead:
            self._log_event("call_refused_after_ttl", op="write_file", path=sandbox_path)
            raise SandboxUnavailable(
                f"sandbox {self.sandbox_id} instance was recycled (TTL) and is not reachable"
            )
        os.makedirs(os.path.dirname(sandbox_path), exist_ok=True)
        if isinstance(content, Path):
            shutil.copyfile(content, sandbox_path)
        elif isinstance(content, bytes):
            with open(sandbox_path, "wb") as f:
                f.write(content)
        else:
            with open(sandbox_path, "w") as f:
                f.write(content)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        if self._dead:
            self._log_event("call_refused_after_ttl", op="read_file", path=sandbox_path)
            raise SandboxUnavailable(
                f"sandbox {self.sandbox_id} instance was recycled (TTL) and is not reachable"
            )
        with open(sandbox_path) as f:
            return f.read()

    # -- local-backend extensions --------------------------------------------

    async def instance_workdir(self, workdir: str) -> str:
        """Materialize the task instance for this episode and return its path.

        `workdir` is the instance template path (relative paths resolve
        against SLIME_INSTANCE_STORE). The template is copied into the
        sandbox root so every episode starts from the pristine base state.
        """
        template = workdir
        if not os.path.isabs(template):
            store = os.environ.get(INSTANCE_STORE_ENV, "")
            template = os.path.join(store, workdir) if store else os.path.abspath(workdir)
        if not os.path.isdir(template):
            raise FileNotFoundError(f"instance template not found: {template}")
        dst = os.path.join(self.root, "repo")
        shutil.copytree(template, dst, symlinks=True)
        self._log_event("instance_materialized", template=template, workdir=dst)
        return dst

    def snapshot_episode(self, ep_dir: str) -> None:
        """Copy the episode's sandbox-side artifacts into the archive dir.

        Host-filesystem copy: works even after the instance was recycled
        (killed processes do not delete files), which is exactly what the
        post-hoc audit needs.
        """
        os.makedirs(ep_dir, exist_ok=True)
        repo = os.path.join(self.root, "repo")
        if os.path.isdir(repo):
            dest = os.path.join(ep_dir, "sandbox_workdir")
            if not os.path.exists(dest):
                shutil.copytree(
                    repo,
                    dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    symlinks=True,
                )

    def cleanup_root(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- helpers -------------------------------------------------------------

    def _kill_pid(self, pid: int) -> None:
        try:
            os.kill(-pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["HOME"] = self.home
        env["SBX_HOME"] = self.home
        # the adapter and every in-sandbox service are loopback; never let a
        # corporate proxy intercept them
        env["no_proxy"] = "127.0.0.1,localhost"
        env["NO_PROXY"] = env["no_proxy"]
        return env

    def _log_event(self, event: str, **fields: Any) -> None:
        rec = {"ts": round(time.time(), 3), "sandbox_id": self.sandbox_id, "event": event}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False)
        if self.ep_dir:
            try:
                os.makedirs(self.ep_dir, exist_ok=True)
                with open(os.path.join(self.ep_dir, "sandbox.log"), "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
