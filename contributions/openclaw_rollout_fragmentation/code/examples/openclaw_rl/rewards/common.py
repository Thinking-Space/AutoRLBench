"""Shared grouping and task-advantage helpers for OpenClaw reward designs."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

import torch

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample


OPTIONAL_ADVISOR_PROMPT = """

You have access to a `cloud_assist` tool backed by a stronger reviewer model. It takes NO parameters: when you call cloud_assist(), your current conversation history is forwarded and the advice returns as tool context.

Cloud assistance is a limited external resource. Call it only when you expect the advice to materially improve the final task result compared with continuing locally. Good reasons include a consequential ambiguity, a repeatedly failing approach, missing expertise, or a final verification where a likely error remains. Do not call it as a routine opening step or merely for reassurance. If local evidence is sufficient, complete the task without cloud assistance.
"""


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


def advisor_telemetry_metadata(advisor_stats: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize advisor audit aggregates into per-rollout metadata."""
    stats = advisor_stats or {}
    return {
        "advisor_calls": int(stats.get("calls", 0) or 0),
        "advisor_successful_calls": int(stats.get("successful_calls", 0) or 0),
        "advisor_failed_calls": int(stats.get("failed_calls", 0) or 0),
        "advisor_cloud_attempts": int(stats.get("cloud_attempts", 0) or 0),
        "advisor_cloud_retries": int(stats.get("cloud_retries", 0) or 0),
        "advisor_cloud_latency_ms": float(stats.get("cloud_latency_ms", 0.0) or 0.0),
        "advisor_cloud_max_latency_ms": float(stats.get("cloud_max_latency_ms", 0.0) or 0.0),
        "advisor_cloud_failure_status_counts": dict(stats.get("failure_status_counts") or {}),
        "advisor_cloud_tokens": int(stats.get("total_tokens", 0) or 0),
        "advisor_cloud_cost_rmb": float(stats.get("cost_rmb", 0.0) or 0.0),
        "advisor_cloud_model": stats.get("cloud_model", ""),
        "advisor_cost_pricing_rmb_per_1k": stats.get("pricing_rmb_per_1k", {}),
        "advisor_tool_invocations": int(stats.get("tool_invocations", 0) or 0),
        "advisor_tool_returns": int(stats.get("tool_returns", 0) or 0),
        "advisor_pending_tool_calls": int(stats.get("pending_tool_calls", 0) or 0),
        "advisor_cloud_responses": int(stats.get("cloud_responses", 0) or 0),
        "advisor_cloud_response_without_tool_return": int(
            stats.get("cloud_response_without_tool_return", 0) or 0
        ),
        "advisor_trace_attempts": int(stats.get("trace_attempts", 0) or 0),
        "advisor_trace_attempt_errors": int(stats.get("trace_attempt_errors", 0) or 0),
        "advisor_trace_attempt_timeouts": int(stats.get("trace_attempt_timeouts", 0) or 0),
        "advisor_transcript_tool_calls": int(stats.get("transcript_tool_calls", 0) or 0),
        "advisor_transcript_tool_results": int(stats.get("transcript_tool_results", 0) or 0),
        "advisor_transcript_pending_tool_calls": int(
            stats.get("transcript_pending_tool_calls", 0) or 0
        ),
        "advisor_committed_tool_result_calls": int(
            stats.get("committed_tool_result_calls", 0) or 0
        ),
        "advisor_tool_return_without_committed_result": int(
            stats.get("tool_return_without_committed_result", 0) or 0
        ),
        "advisor_model_requests": int(stats.get("model_requests", 0) or 0),
        "advisor_model_responses": int(stats.get("model_responses", 0) or 0),
        "advisor_followup_model_requests": int(stats.get("followup_model_requests", 0) or 0),
        "advisor_followup_model_responses": int(stats.get("followup_model_responses", 0) or 0),
        "advisor_transcript_tool_results_without_followup": int(
            stats.get("transcript_tool_results_without_followup", 0) or 0
        ),
    }


def record_is_excluded(record: RolloutRecord) -> bool:
    """True when any Sample of the rollout was aborted or marked for removal."""
    return any(
        sample.remove_sample or sample.status == Sample.Status.ABORTED
        for _position, sample in record.indexed_samples
    )


def agent_behavior_metrics(
    records: list[RolloutRecord], *, include_advisor_breakdown: bool = True
) -> dict[str, float]:
    """Batch-level agent behavior aggregates from per-rollout metadata.

    Reward-mode agnostic: any ``post_process_rewards`` that reports metrics can
    merge these into its log payload. ``records`` should already exclude
    aborted rollouts so the averages describe real trajectories.
    """
    rollout_count = len(records)
    total_turns = sum(int(record.metadata.get("agent_turns", 0) or 0) for record in records)
    total_tool_calls = sum(int(record.metadata.get("agent_tool_calls", 0) or 0) for record in records)
    fragments = [len(record.indexed_samples) for record in records]
    generated_tokens = [int(record.metadata.get("agent_generated_tokens", 0) or 0) for record in records]
    max_context_tokens = [int(record.metadata.get("agent_max_context_tokens", 0) or 0) for record in records]
    episode_elapsed_sec = [
        float(record.metadata["sandbox_elapsed_sec"])
        for record in records
        if isinstance(record.metadata.get("sandbox_elapsed_sec"), (int, float))
        and float(record.metadata["sandbox_elapsed_sec"]) >= 0
        and math.isfinite(float(record.metadata["sandbox_elapsed_sec"]))
    ]
    trained_response_tokens = [
        sum(
            sum(sample.loss_mask) if sample.loss_mask is not None else int(sample.response_length or 0)
            for _position, sample in record.indexed_samples
        )
        for record in records
    ]
    response_region_tokens = [
        sum(int(sample.response_length or 0) for _position, sample in record.indexed_samples)
        for record in records
    ]
    task_rewards = [record.task_reward for record in records]
    # Tokens of a trailing tool-call turn whose result never arrived; those get
    # loss_mask=0 in generate.py because their consequence is unobservable.
    dangling_masked_tokens = [
        int(record.metadata.get("dangling_final_turn_masked_tokens", 0) or 0) for record in records
    ]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    metrics = {
        "openclaw/agent_turns_per_rollout": total_turns / rollout_count if rollout_count else 0.0,
        "openclaw/agent_tool_calls_per_rollout": total_tool_calls / rollout_count if rollout_count else 0.0,
        "openclaw/agent_tool_calls_per_turn": total_tool_calls / total_turns if total_turns else 0.0,
        "openclaw/episode_elapsed_sec_mean": mean(episode_elapsed_sec),
        "openclaw/episode_elapsed_sec_p50": percentile(episode_elapsed_sec, 0.50),
        "openclaw/episode_elapsed_sec_p90": percentile(episode_elapsed_sec, 0.90),
        "openclaw/episode_elapsed_sec_p95": percentile(episode_elapsed_sec, 0.95),
        "openclaw/episode_elapsed_sec_p99": percentile(episode_elapsed_sec, 0.99),
        "openclaw/episode_elapsed_sec_max": max(episode_elapsed_sec, default=0.0),
        # These are episode-level metrics. Unlike rollout/total_lengths, they do
        # not average the fragments emitted when token replay drift forks a run.
        "openclaw/fragments_per_rollout": mean(fragments),
        "openclaw/forked_rollout_rate": (
            sum(fragment_count > 1 for fragment_count in fragments) / rollout_count if rollout_count else 0.0
        ),
        "openclaw/episode_generated_tokens_mean": mean(generated_tokens),
        "openclaw/episode_generated_tokens_p50": float(median(generated_tokens)) if generated_tokens else 0.0,
        "openclaw/episode_generated_tokens_max": max(generated_tokens, default=0),
        "openclaw/episode_max_context_tokens_mean": mean(max_context_tokens),
        "openclaw/episode_max_context_tokens_p50": float(median(max_context_tokens)) if max_context_tokens else 0.0,
        "openclaw/episode_max_context_tokens_max": max(max_context_tokens, default=0),
        "openclaw/episode_trained_response_tokens_mean": mean(trained_response_tokens),
        "openclaw/episode_trained_response_tokens_p50": (
            float(median(trained_response_tokens)) if trained_response_tokens else 0.0
        ),
        "openclaw/episode_trained_response_tokens_max": max(trained_response_tokens, default=0),
        "openclaw/episode_response_region_tokens_mean": mean(response_region_tokens),
        "openclaw/episode_reward_mean": mean(task_rewards),
        "openclaw/episode_reward_p50": float(median(task_rewards)) if task_rewards else 0.0,
        "openclaw/episode_reward_max": max(task_rewards, default=0.0),
        "openclaw/episode_trained_token_ratio": (
            sum(trained_response_tokens) / sum(generated_tokens) if sum(generated_tokens) else 0.0
        ),
        "openclaw/episode_replay_inflation_ratio": (
            sum(response_region_tokens) / sum(generated_tokens) if sum(generated_tokens) else 0.0
        ),
        "openclaw/dangling_final_turn_rate": (
            sum(1 for tokens in dangling_masked_tokens if tokens > 0) / rollout_count if rollout_count else 0.0
        ),
        "openclaw/dangling_masked_tokens_mean": mean(dangling_masked_tokens),
        "openclaw/dangling_masked_tokens_total": sum(dangling_masked_tokens),
    }

    if include_advisor_breakdown:
        call_buckets = {
            "0": lambda calls: calls == 0,
            "1": lambda calls: calls == 1,
            "2": lambda calls: calls == 2,
            "3plus": lambda calls: calls >= 3,
        }
        for label, includes in call_buckets.items():
            positions = [
                position
                for position, record in enumerate(records)
                if includes(int(record.metadata.get("advisor_calls", 0) or 0))
            ]
            metrics[f"openclaw/advisor_calls_{label}_rollouts"] = len(positions)
            metrics[f"openclaw/advisor_calls_{label}_generated_tokens_mean"] = mean(
                [generated_tokens[position] for position in positions]
            )
            metrics[f"openclaw/advisor_calls_{label}_trained_response_tokens_mean"] = mean(
                [trained_response_tokens[position] for position in positions]
            )
            metrics[f"openclaw/advisor_calls_{label}_max_context_tokens_mean"] = mean(
                [max_context_tokens[position] for position in positions]
            )
    return metrics


_behavior_log_count = 0


def _advisor_metrics_enabled(records: list[RolloutRecord]) -> bool:
    override = os.environ.get("OPENCLAW_SWANLAB_ADVISOR_METRICS", "auto").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    if os.environ.get("OPENCLAW_ADVISOR", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    advisor_keys = (
        "advisor_calls",
        "advisor_cloud_attempts",
        "advisor_tool_invocations",
        "advisor_model_requests",
    )
    return any(any(record.metadata.get(key) for key in advisor_keys) for record in records)


def log_agent_behavior_metrics(args, records: list[RolloutRecord]) -> None:
    """Report agent behavior metrics for reward modes without their own telemetry.

    Aborted rollouts are filtered out here. The step axis uses a module-local
    counter that is NOT checkpointed, so after a resume the curve restarts at
    step 0. Reward modes with checkpointed telemetry (counterfactual_cost_budget)
    should merge :func:`agent_behavior_metrics` into their own log payload
    instead of calling this.
    """
    global _behavior_log_count
    if not (
        getattr(args, "use_wandb", False)
        or getattr(args, "use_tensorboard", False)
        or getattr(args, "use_swanlab", False)
    ):
        return
    active_records = [record for record in records if not record_is_excluded(record)]
    include_advisor = _advisor_metrics_enabled(active_records)
    metrics = agent_behavior_metrics(active_records, include_advisor_breakdown=include_advisor)
    rollout_count = len(active_records)

    def total(key: str) -> int:
        return sum(int(record.metadata.get(key, 0) or 0) for record in active_records)

    advisor_fields = {
        "advisor/cloud_calls": "advisor_calls",
        "advisor/cloud_successful_calls": "advisor_successful_calls",
        "advisor/cloud_failed_calls": "advisor_failed_calls",
        "advisor/cloud_attempts": "advisor_cloud_attempts",
        "advisor/cloud_retry_count": "advisor_cloud_retries",
        "advisor/tool_invocations": "advisor_tool_invocations",
        "advisor/tool_returns": "advisor_tool_returns",
        "advisor/pending_tool_calls": "advisor_pending_tool_calls",
        "advisor/cloud_responses": "advisor_cloud_responses",
        "advisor/cloud_response_without_tool_return": "advisor_cloud_response_without_tool_return",
        "advisor/trace_attempts": "advisor_trace_attempts",
        "advisor/trace_attempt_errors": "advisor_trace_attempt_errors",
        "advisor/trace_attempt_timeouts": "advisor_trace_attempt_timeouts",
        "advisor/transcript_tool_calls": "advisor_transcript_tool_calls",
        "advisor/transcript_tool_results": "advisor_transcript_tool_results",
        "advisor/transcript_pending_tool_calls": "advisor_transcript_pending_tool_calls",
        "advisor/committed_tool_result_calls": "advisor_committed_tool_result_calls",
        "advisor/tool_return_without_committed_result": "advisor_tool_return_without_committed_result",
        "advisor/model_requests": "advisor_model_requests",
        "advisor/model_responses": "advisor_model_responses",
        "advisor/followup_model_requests": "advisor_followup_model_requests",
        "advisor/followup_model_responses": "advisor_followup_model_responses",
        "advisor/transcript_tool_results_without_followup": "advisor_transcript_tool_results_without_followup",
    }
    if include_advisor:
        for metric_name, metadata_key in advisor_fields.items():
            value = total(metadata_key)
            metrics[metric_name] = value
            metrics[f"{metric_name}_per_rollout"] = value / rollout_count if rollout_count else 0.0
        metrics["advisor/cloud_success_rate"] = (
            total("advisor_successful_calls") / total("advisor_calls") if total("advisor_calls") else 0.0
        )
    metrics["rollout/step"] = compute_rollout_step(args, _behavior_log_count)
    _behavior_log_count += 1
    logging_utils.log(args, metrics, step_key="rollout/step")
