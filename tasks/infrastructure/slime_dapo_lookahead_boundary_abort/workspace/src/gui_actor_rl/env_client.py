"""Env client adapter for GUI actor RL.

Wraps guirl's AsyncRpcEnvClient (env master :9900 + docker-VM rpc workers)
behind the interface the gui-rl rollout loop expects (GuiEnvClient), namely:

- ``step(lease_id, action, sleep_after_execution)`` — guirl's step() has no
  sleep parameter, so we sleep client-side and re-fetch the observation.
- ``start_recording`` / ``end_recording`` — not supported by the guirl env
  service; kept as no-ops so the rollout code stays close to upstream gui-rl.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from guirl.env_service.rpc_env_client import AsyncRpcEnvClient, RpcEnvError

logger = logging.getLogger(__name__)
_PROCESS_OWNER_ID = f"gui-actor-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class ActorEnvClient:
    """gui-rl compatible facade over guirl's AsyncRpcEnvClient."""

    def __init__(self, base_url: str):
        self._client = AsyncRpcEnvClient(base_url)
        self.owner_id = os.getenv("GUI_ENV_OWNER", "").strip() or _PROCESS_OWNER_ID
        self.owner = {"component": "gui-actor-rl", "run_id": self.owner_id}
        self._fence_tokens: dict[str, str] = {}
        self._allocations: dict[str, str] = {}
        self._lease_episodes: dict[str, str] = {}
        # The guirl rpc worker's OSWorldEnv already sleeps `sleep_after_execution`
        # (2.0 s) before taking the post-action screenshot. With this flag set the
        # client trusts that and skips its own sleep + second screenshot per step
        # (on AGS a screenshot round trip is the dominant per-step cost).
        self._worker_sleeps = os.getenv("GUI_ENV_WORKER_SLEEPS_AFTER_EXECUTION", "0").strip().lower() in {"1", "true", "yes"}

    async def allocate(self, episode_id: str) -> dict[str, Any]:
        # Retain the caller-chosen fence even if the allocate response is lost.
        fence = self._allocations.setdefault(episode_id, uuid.uuid4().hex)
        try:
            lease = await self._client.allocate(
                episode_id,
                allocation_key=f"{self.owner_id}:{episode_id}",
                owner=self.owner,
                fence_token=fence,
                generation=1,
            )
        except RpcEnvError as exc:
            # These responses are authoritative rejections: this request did
            # not create an unknown remote lease, so retaining its local fence
            # would only grow bookkeeping until the whole session shuts down.
            # Transport failures and allocation_pending remain retained because
            # the server may have accepted the request before the response was
            # lost; close_session() must cancel those with the same fence.
            if exc.code in {
                "validation_error",
                "episode_conflict",
                "idempotency_conflict",
                "fenced",
                "fence_required",
                "allocation_cancelled",
            }:
                self._allocations.pop(episode_id, None)
            raise
        lease_id = str(lease["lease_id"])
        fence_token = lease.get("fence_token")
        if not fence_token:
            raise RuntimeError(f"env master returned no fence token for lease {lease_id}")
        self._fence_tokens[lease_id] = str(fence_token)
        self._lease_episodes[lease_id] = episode_id
        return lease

    def fence_token(self, lease_id: str) -> str | None:
        return getattr(self, "_fence_tokens", {}).get(lease_id)

    async def heartbeat(self, lease_id: str) -> None:
        await self._client.heartbeat(lease_id, fence_token=self.fence_token(lease_id))

    @staticmethod
    def _op_id(lease_id: str, kind: str) -> str:
        # One id per LOGICAL call, reused across AsyncRpcEnvClient's transport
        # retries, so a re-sent /step or /reset after a dropped connection is
        # deduplicated by the worker instead of executing the action twice.
        return f"{lease_id}:{kind}:{uuid.uuid4().hex}"

    async def reset(self, lease_id: str, task_config: dict[str, Any] | None) -> dict[str, Any]:
        return await self._client.reset(
            lease_id,
            task_config,
            fence_token=self.fence_token(lease_id),
            operation_id=self._op_id(lease_id, "reset"),
        )

    async def get_obs(self, lease_id: str) -> dict[str, Any]:
        return await self._client.get_obs(
            lease_id,
            fence_token=self.fence_token(lease_id),
            operation_id=self._op_id(lease_id, "get_obs"),
        )

    async def step(
        self, lease_id: str, action: Any, sleep_after_execution: float = 0.0
    ) -> tuple[dict[str, Any], float, bool, dict]:
        fence_token = self.fence_token(lease_id)
        obs, reward, done, info = await self._client.step(
            lease_id,
            str(action),
            fence_token=fence_token,
            operation_id=self._op_id(lease_id, "step"),
        )
        if sleep_after_execution and sleep_after_execution > 0 and not self._worker_sleeps:
            await asyncio.sleep(float(sleep_after_execution))
            try:
                obs = await self._client.get_obs(
                    lease_id,
                    fence_token=fence_token,
                    operation_id=self._op_id(lease_id, "get_obs"),
                )
            except Exception:
                logger.warning("get_obs after post-step sleep failed for lease %s", lease_id, exc_info=True)
        return obs, reward, done, info

    async def evaluate(self, lease_id: str) -> float:
        return await self._client.evaluate(
            lease_id,
            fence_token=self.fence_token(lease_id),
            operation_id=self._op_id(lease_id, "evaluate"),
        )

    async def start_recording(self, lease_id: str) -> None:
        # guirl env service has no recording endpoint; keep the rollout code intact.
        return None

    async def end_recording(self, lease_id: str, out_path: str) -> None:
        return None

    async def close(self, lease_id: str) -> None:
        await self._client.close_strict(
            lease_id, fence_token=self.fence_token(lease_id)
        )
        getattr(self, "_fence_tokens", {}).pop(lease_id, None)
        episode = self._lease_episodes.pop(lease_id, None)
        self._allocations.pop(episode, None)

    async def close_session(self) -> bool:
        """Return True only after all owned allocations have cleanup acknowledgments.

        A timed-out /close can still finish remotely. Cancellation with the
        original allocation key and fence resolves that uncertain outcome;
        closing the HTTP transport alone is never a cleanup acknowledgment.
        """
        async def cancel_remaining():
            # A lost allocate response or failed close must still be fenced at
            # the master. Pending worker allocation can return a retryable 409.
            while self._allocations:
                for episode, fence in list(self._allocations.items()):
                    try:
                        await self._client.cancel_allocation(
                            f"{self.owner_id}:{episode}",
                            operation_id=f"cancel:{self.owner_id}:{episode}",
                            fence_token=fence,
                            generation=1,
                        )
                    except Exception:
                        logger.warning("Allocation cleanup failed for %s; retrying", episode, exc_info=True)
                    else:
                        self._allocations.pop(episode, None)
                if self._allocations:
                    await asyncio.sleep(1)

        try:
            if self._allocations:
                await asyncio.wait_for(
                    cancel_remaining(),
                    timeout=float(os.getenv("GUI_ENV_CLOSE_TIMEOUT_S", "120")),
                )
        finally:
            await self._client.close_session()
            if not self._allocations:
                self._fence_tokens.clear()
                self._lease_episodes.clear()
            else:
                logger.error("Env cleanup incomplete; outstanding allocation keys: %s",
                             [f"{self.owner_id}:{episode}" for episode in self._allocations])
        return True
