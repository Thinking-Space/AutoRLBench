"""Ray actors that shard E2B transports across worker processes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from slime.agent.sandbox import E2BSandbox, ExecResult, FileContent

logger = logging.getLogger(__name__)


class _E2BSandboxActor:
    """Own E2B SDK objects inside one process and event loop."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, E2BSandbox] = {}

    async def create(self, key: str, image: str, kwargs: dict[str, Any]) -> str:
        if key in self._sandboxes:
            raise RuntimeError(f"sandbox key already exists: {key}")
        sandbox = E2BSandbox(image, **kwargs)
        await sandbox.__aenter__()
        self._sandboxes[key] = sandbox
        return sandbox.sandbox_id

    def _get(self, key: str) -> E2BSandbox:
        try:
            return self._sandboxes[key]
        except KeyError:
            raise RuntimeError(f"sandbox key is not active: {key}") from None

    async def exec(
        self,
        key: str,
        cmd: str,
        *,
        user: str,
        env: dict[str, str] | None,
        timeout: int,
        check: bool,
        idempotent: bool,
    ) -> ExecResult:
        return await self._get(key).exec(
            cmd,
            user=user,
            env=env,
            timeout=timeout,
            check=check,
            idempotent=idempotent,
        )

    async def write_file(self, key: str, sandbox_path: str, content: str | bytes, *, user: str) -> None:
        await self._get(key).write_file(sandbox_path, content, user=user)

    async def read_file(self, key: str, sandbox_path: str, *, user: str) -> str:
        return await self._get(key).read_file(sandbox_path, user=user)

    async def close(self, key: str) -> None:
        sandbox = self._sandboxes.pop(key, None)
        if sandbox is not None:
            await sandbox.__aexit__(None, None, None)


class RayE2BSandbox:
    """Sandbox protocol proxy backed by one actor from a sharded pool."""

    def __init__(self, actor, image: str, kwargs: dict[str, Any]) -> None:
        self._actor = actor
        self._image = image
        self._kwargs = kwargs
        self._key = uuid.uuid4().hex
        self._entered = False
        self.sandbox_id = ""

    async def __aenter__(self) -> RayE2BSandbox:
        self.sandbox_id = await self._actor.create.remote(self._key, self._image, self._kwargs)
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._entered:
            return
        self._entered = False
        try:
            await self._actor.close.remote(self._key)
        except Exception as close_error:
            logger.warning("[openclaw-e2b-actor] close %s failed: %s", self.sandbox_id[:8], close_error)

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
        return await self._actor.exec.remote(
            self._key,
            cmd,
            user=user,
            env=env,
            timeout=timeout,
            check=check,
            idempotent=idempotent,
        )

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        # Ray workers need not share the caller's filesystem mount. Materialize
        # host paths before crossing the actor boundary.
        if isinstance(content, Path):
            content = content.read_bytes()
        await self._actor.write_file.remote(self._key, sandbox_path, content, user=user)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._actor.read_file.remote(self._key, sandbox_path, user=user)


class RayE2BSandboxPool:
    """Distribute new E2B sandboxes evenly across isolated actor shards."""

    def __init__(self, actor_count: int, max_concurrency: int) -> None:
        if actor_count < 1:
            raise ValueError("E2B actor count must be positive")
        if max_concurrency < 1:
            raise ValueError("E2B actor max concurrency must be positive")

        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before creating E2B sandbox actors")
        actor_cls = ray.remote(_E2BSandboxActor)
        self._actors = [
            actor_cls.options(num_cpus=0, max_concurrency=max_concurrency).remote() for _ in range(actor_count)
        ]
        self._next_actor = 0

    def sandbox(self, session_id: str, image: str, **kwargs: Any) -> RayE2BSandbox:
        del session_id
        actor = self._actors[self._next_actor]
        self._next_actor = (self._next_actor + 1) % len(self._actors)
        return RayE2BSandbox(actor, image, kwargs)
