"""Native synchronous rollout adapter for cached teacher trajectories."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

from opqad.integrations.teacher_replay_rollout import ENV_VAR, TeacherReplayRollout


class ReplayRollout(BaseRollout):
    """Serve cached responses through the normal FSDP rollout interface."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        trajectory_path = os.environ.get(ENV_VAR, "").strip()
        if not trajectory_path:
            raise ValueError(f"{ENV_VAR} must point to a teacher trajectory parquet for rollout.name=replay")

        adapter_config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=config,
                model=SimpleNamespace(path=model_config.path),
            )
        )
        self._replay = TeacherReplayRollout(
            trajectory_path=trajectory_path,
            config=adapter_config,
            rollout_wg=None,
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        return self._replay.replay(prompts)

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        return None

    async def release(self):
        return None
