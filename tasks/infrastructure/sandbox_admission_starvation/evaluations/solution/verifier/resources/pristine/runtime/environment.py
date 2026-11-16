"""Per-trajectory tool environment backed by a remote sandbox."""
from __future__ import annotations

import httpx

from agent_runtime import env_runtime


class EnvironmentError_(RuntimeError):
    """Raised when the sandbox cannot serve a tool call."""


class _Env:
    def __init__(self, env: env_runtime.SandboxEnv):
        self._env = env

    async def call(self, name: str, params: dict) -> str:
        try:
            return await self._env.call(name, params)
        except httpx.HTTPError as exc:
            raise EnvironmentError_(f"tool {name} failed: {exc!r}") from exc

    async def close(self) -> None:
        await self._env.close()


async def open_environment(sample) -> _Env:
    metadata = sample.metadata or {}
    session = sample.routing_key or f"{metadata.get('task_id')}-{sample.index}"
    env = await env_runtime.create_and_setup_env(session, metadata["files"], metadata["test_failure"])
    return _Env(env)
