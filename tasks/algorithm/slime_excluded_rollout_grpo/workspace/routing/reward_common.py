"""Shared grouping and task-advantage helpers for OpenClaw reward designs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from slime.utils.types import Sample


@dataclass
class RolloutRecord:
    """One terminal rollout, which may fan out into multiple training Samples."""

    key: tuple[Any, Any]
    indexed_samples: list[tuple[int, Sample]]
    task_reward: float
    metadata: dict[str, Any]


def collect_rollouts(args, samples: list[Sample]) -> tuple[list[RolloutRecord], dict[Any, list[RolloutRecord]]]:
    """Collapse fan-out Samples and regroup terminal rollouts by prompt."""
    grouped_samples: dict[tuple[Any, Any], list[tuple[int, Sample]]] = defaultdict(list)
    for position, sample in enumerate(samples):
        rollout_key = sample.rollout_id if sample.rollout_id is not None else sample.index
        grouped_samples[(sample.group_index, rollout_key)].append((position, sample))

    records: list[RolloutRecord] = []
    prompt_groups: dict[Any, list[RolloutRecord]] = defaultdict(list)
    for key, indexed_samples in grouped_samples.items():
        metadata = next((sample.metadata for _, sample in indexed_samples if sample.metadata), {}) or {}
        metadata_rewards = [
            float(sample.metadata["raw_reward"])
            for _, sample in indexed_samples
            if sample.metadata and "raw_reward" in sample.metadata
        ]
        task_reward = (
            metadata_rewards[0]
            if metadata_rewards
            else sum(float(sample.get_reward_value(args)) for _, sample in indexed_samples)
        )
        record = RolloutRecord(
            key=key,
            indexed_samples=indexed_samples,
            task_reward=task_reward,
            metadata=dict(metadata),
        )
        records.append(record)
        prompt_groups[key[0]].append(record)
    return records, prompt_groups


def task_advantages(args, prompt_records: list[RolloutRecord]) -> dict[tuple[Any, Any], float]:
    """Compute the repository's usual group-relative task advantages."""
    values = torch.tensor([record.task_reward for record in prompt_records], dtype=torch.float32)
    normalized = values
    normalize = (
        args.advantage_estimator in {"grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"}
        and args.rewards_normalization
    )
    if normalize:
        normalized = values - values.mean()
        if args.advantage_estimator in {"grpo", "gspo", "cispo"} and args.grpo_std_normalization:
            normalized = normalized / (normalized.std() + 1e-6) if len(values) > 1 else torch.zeros_like(normalized)
    return {
        record.key: float(advantage) for record, advantage in zip(prompt_records, normalized.tolist(), strict=True)
    }


def expand_rollout_values(
    samples: list[Sample],
    records: list[RolloutRecord],
    values: dict[tuple[Any, Any], float],
) -> list[float]:
    expanded = [0.0] * len(samples)
    for record in records:
        value = values[record.key]
        for position, _sample in record.indexed_samples:
            expanded[position] = value
    return expanded
