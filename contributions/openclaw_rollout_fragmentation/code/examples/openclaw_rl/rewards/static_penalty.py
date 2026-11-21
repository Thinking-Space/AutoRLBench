"""Legacy fixed advisor-call penalty.

Design
------
This module preserves the first Advisor-OpenClaw reward design exactly:

    penalty = min(penalty_per_call * advisor_calls, penalty_cap)
    reward = max(0, clawgym_reward - penalty)

The penalized terminal reward is then normalized within each GRPO prompt group.
It is intentionally kept as a separate selectable baseline. It does not attempt
to determine whether a call was necessary, and calls beyond the cap have no
additional marginal cost.
"""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from .common import (
    advisor_telemetry_metadata,
    collect_rollouts,
    expand_rollout_values,
    log_agent_behavior_metrics,
    task_advantages,
)


def rollout_role(_args, _sample: Sample) -> str:
    return "policy"


def advisor_prompt(default_prompt: str) -> str:
    return default_prompt


def finalize_rollout(
    clawgym_reward: float,
    advisor_stats: dict[str, Any] | None,
    *,
    penalty_per_call: float,
    penalty_cap: float,
    role: str,
) -> tuple[float, dict[str, Any]]:
    del role
    if advisor_stats is None:
        return clawgym_reward, {"raw_reward": clawgym_reward}
    penalty = min(penalty_per_call * advisor_stats["calls"], penalty_cap)
    reward = max(0.0, clawgym_reward - penalty)
    return reward, {
        "raw_reward": reward,
        "clawgym_reward": clawgym_reward,
        **advisor_telemetry_metadata(advisor_stats),
        "advisor_penalty": penalty,
        "advisor_reward_mode": "static_penalty",
    }


def post_process_rewards(args, samples: list[Sample]):
    records, prompt_groups = collect_rollouts(args, samples)
    log_agent_behavior_metrics(args, records)
    raw_by_key = {record.key: record.task_reward for record in records}
    advantage_by_key: dict[tuple[Any, Any], float] = {}
    for prompt_records in prompt_groups.values():
        advantage_by_key.update(task_advantages(args, prompt_records))
    return (
        expand_rollout_values(samples, records, raw_by_key),
        expand_rollout_values(samples, records, advantage_by_key),
    )
