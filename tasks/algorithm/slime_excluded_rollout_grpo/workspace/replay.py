#!/usr/bin/env python3
"""Replay the task-advantage path with synthetic post-rollout records."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from reward_common import collect_rollouts, expand_rollout_values, task_advantages


@dataclass
class Sample:
    """Only the historical helper's inputs and the trainer exclusion flags."""

    class Status(Enum):
        COMPLETED = "completed"
        ABORTED = "aborted"

    index: int
    group_index: int
    rollout_id: str
    metadata: dict[str, float]
    remove_sample: bool
    status: Status


def read_samples(rows):
    samples = []
    for row in rows:
        for _ in range(row.get("fragments", 1)):
            samples.append(Sample(
                index=len(samples),
                group_index=row.get("group_index", 0),
                rollout_id=row["rollout_id"],
                metadata={"raw_reward": row["reward"]},
                remove_sample=row.get("remove_sample", False),
                status=Sample.Status(row.get("status", "completed")),
            ))
    return samples


def post_process_rewards(args, samples):
    # Reduced from the historical cost-budget callback. This is its task-side
    # call sequence. Cloud cost and dual updates are outside this replay.
    records, groups = collect_rollouts(args, samples)
    by_rollout = {}
    for prompt_records in groups.values():
        by_rollout.update(task_advantages(args, prompt_records))
    return expand_rollout_values(samples, records, by_rollout)


def run_batch(rows):
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    samples = read_samples(rows)
    advantages = post_process_rewards(args, samples)
    fragments = []
    for sample, advantage in zip(samples, advantages, strict=True):
        trainable = not (sample.remove_sample or sample.status == Sample.Status.ABORTED)
        fragments.append({
            "rollout_id": sample.rollout_id,
            "reward": sample.metadata["raw_reward"],
            "remove_sample": sample.remove_sample,
            "status": sample.status.value,
            "task_advantage": advantage,
            "effective_advantage": advantage if trainable else 0.0,
        })
    return fragments


def main():
    batches = json.loads(Path(__file__).with_name("batches.json").read_text())
    result = {name: run_batch(rows) for name, rows in batches.items()}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
