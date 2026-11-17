import dataclasses
import logging
from typing import Any

import numpy as np

from miles.ray.rollout.batch_padding import is_batch_pad
from miles.rollout.session.metrics import aggregate_session_summaries
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import (
    compute_avg_max_min_percentile_statistics,
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from miles.utils.misc import load_function
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if (x := args.custom_eval_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        num_none = sum(1 for r in rewards if r is None)
        log_dict[f"eval/{key}-none_reward_ratio"] = num_none / len(rewards) if len(rewards) > 0 else 0.0
        if num_none:
            logger.warning(
                f"eval/{key}: {num_none}/{len(rewards)} samples have reward=None "
                "(likely errored/aborted trials); treating as 0.0 for metrics."
            )
            rewards = [0.0 if r is None else r for r in rewards]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if (x := args.custom_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    log_dict |= dict_add_prefix(_compute_gateway_metrics(args, samples), "partial/")
    if getattr(args, "sglang_enable_metrics", False):
        sglang_metrics = _fetch_sglang_metrics(args)
        if sglang_metrics:
            log_dict |= dict_add_prefix(sglang_metrics, "sglang/")
            logger.info(f"sglang metrics {rollout_id}: {sglang_metrics}")
    if args.log_passrate:
        log_dict |= dict_add_prefix(
            _compute_passrate_from_samples(args, samples),
            "passrate/",
        )
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking.log(args, log_dict, step_key="rollout/step")


def _fetch_sglang_metrics(args) -> dict[str, float]:
    """Fetch key Prometheus metrics from sglang engine workers.

    Queries each worker's ``/metrics`` endpoint via the sglang router and
    aggregates into a flat dict. Rates/ratios are averaged across workers;
    counts are summed. No-op (returns empty) when the router is unavailable.
    """
    from miles.utils.tracking_utils.sglang_metrics_utils import fetch_sglang_worker_metrics

    try:
        return fetch_sglang_worker_metrics(args.sglang_router_ip, args.sglang_router_port)
    except Exception as exc:
        logger.warning(f"Failed to fetch sglang worker metrics for rollout log: {exc}")
        return {}


def _compute_metrics_from_samples(args, samples):
    # Batch-alignment padding (fan-out, see batch_padding.py) carries no
    # signal; keep it out of every metric.
    samples = [s for s in samples if not is_batch_pad(s)]
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_session_metrics(samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len(set(s.weight_versions)) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        log_dict["tito_session_mismatch_rate"] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"tito_session_mismatch_rate/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            for strict_type in ("special_token_count", "special_token_type", "non_assistant_text"):
                rate = log_dict.get(f"tito_session_mismatch_rate/{strict_type}", 0)
                assert rate == 0, (
                    f"tito_session_mismatch_rate/{strict_type}={rate:.4f} must be 0 — "
                    "this indicates a bug in the TITO algorithm or chat template. "
                    "Please check your tito model and chat template."
                )
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def _compute_perf_metrics_from_samples(args, samples, rollout_time):
    # Padding is not generated work; excluding it keeps token-throughput honest.
    samples = [s for s in samples if not is_batch_pad(s)]
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


@dataclasses.dataclass
class _TrajectoryReward:
    """One logical rollout slot after collapsing fan-out segments."""

    first_seen: int  # position of the trajectory's first segment, to keep output order stable
    group_index: Any
    reward: float = 0.0


def _collapse_fanout_rewards(args, all_samples: list[Sample]) -> list[tuple[Any, float]]:
    """(group_index, reward) pairs with fan-out trajectories collapsed.

    Black-box fan-out rollouts (examples/black_box_rl*) split one trajectory
    into K segments sharing ``metadata["parent_traj_id"]`` with a sparse
    reward: the anchor segment carries the trajectory's terminal reward, its
    siblings 0. Group diagnostics assume one reward per rollout slot, so a
    per-segment view systematically under-reports zero-std groups and inflates
    group sizes past ``n_samples_per_prompt``. Collapse each trajectory to the
    sum of its segments (== terminal reward). Identity when nothing carries
    ``parent_traj_id`` (non-fan-out runs).
    """
    is_fanout = any((sample.metadata or {}).get("parent_traj_id") is not None for sample in all_samples)
    # Batch-alignment padding has no parent_traj_id; without this filter each
    # padding sample would count as a standalone zero-reward trajectory (and
    # group_index=None padding could even assemble into a phantom all-zero
    # pass@k group when pad_count happens to equal n_samples_per_prompt).
    all_samples = [s for s in all_samples if not is_batch_pad(s)]
    if not is_fanout:
        # An ABORTED sample (e.g. a black-box agent call that errored) carries
        # reward=None; treat it as 0.0 so zero-std/group metrics stay computable
        # instead of crashing the whole rollout on round(None, 1) downstream.
        return [(sample.group_index, float(sample.get_reward_value(args) or 0.0)) for sample in all_samples]

    trajectories: dict[Any, _TrajectoryReward] = {}
    for position, sample in enumerate(all_samples):
        traj_id = (sample.metadata or {}).get("parent_traj_id")
        if traj_id is None:
            # A sample without parent_traj_id (e.g. an aborted placeholder) is
            # its own single-segment trajectory; the position makes the key unique.
            traj_id = ("standalone", position)
        trajectory = trajectories.setdefault(traj_id, _TrajectoryReward(position, sample.group_index))
        trajectory.reward += float(sample.get_reward_value(args) or 0.0)

    ordered = sorted(trajectories.values(), key=lambda t: t.first_seen)
    return [(t.group_index, t.reward) for t in ordered]


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(rewards: list[float]):
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    pairs = _collapse_fanout_rewards(args, all_samples)
    groups_by_index = group_by(pairs, lambda pair: pair[0])
    reward_groups = [[reward for _, reward in group] for group in groups_by_index.values()]
    zero_std_groups = [rewards for rewards in reward_groups if _is_zero_std(rewards)]

    interesting_rewards = [str(round(rewards[0], 1)) for rewards in zero_std_groups]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(reward_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = counts.get("0.0", 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = counts.get("1.0", 0) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_session_metrics(all_samples: list[Sample]) -> dict[str, float]:
    """Session-server counters (turns, tool calls, branches, drift) for this step.

    Empty unless the session server ran: only ``agentic_tool_call`` merges
    ``metadata["session_metrics"]`` onto its samples.

    Deduped by session ID because black-box fan-out emits K segment samples per
    trajectory that all carry the same session metadata — counting per segment
    would weight a K-branch trajectory K times against a linear one, exactly the
    skew ``_collapse_fanout_rewards`` corrects for rewards.
    """
    summaries: dict[str, dict] = {}
    for sample in all_samples:
        summary = (sample.metadata or {}).get("session_metrics")
        if isinstance(summary, dict) and summary.get("session_id"):
            summaries.setdefault(summary["session_id"], summary)
    return dict_add_prefix(aggregate_session_summaries(list(summaries.values())), "session/")


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_gateway_metrics(args, all_samples: list[Sample]):
    """Partial-rollout (gateway EngineClient) abort->resume observability.

    ``gateway_segments_total``/``gateway_resumed_total`` are only populated on
    samples that went through the gateway (see
    ``gateway.client_sdk._attach_gateway_observability``); for a non-gateway
    rollout every sample has 0 segments, so this block silently returns
    nothing rather than logging an all-zero metric group. Caller prefixes the
    result with ``partial/`` so it shows up as its own SwanLab/Prometheus
    group, alongside ``rollout``/``perf``/``train``.
    """
    segments_totals = [s.gateway_segments_total for s in all_samples]
    if not any(segments_totals):
        return {}

    resumed_totals = [s.gateway_resumed_total for s in all_samples]
    num_samples = len(all_samples)
    metrics = dict_add_prefix(compute_statistics(segments_totals), "segments_total/")
    metrics["segments_total"] = sum(segments_totals)
    metrics["resumed_total"] = sum(resumed_totals)
    metrics["resumed_sample_ratio"] = sum(1 for r in resumed_totals if r > 0) / num_samples
    metrics["avg_segments_per_sample"] = sum(segments_totals) / num_samples

    # processing_time/{avg,max,min,tp50,tp95,tp99}: per-sample total engine
    # round-trip time (all segments/resends included). verl's fully_async
    # equivalent (recipe/fully_async_policy/detach_utils.py::processing_time_stats).
    processing_times = [s.gateway_processing_time for s in all_samples if s.gateway_processing_time > 0]
    if processing_times:
        metrics |= dict_add_prefix(compute_avg_max_min_percentile_statistics(processing_times), "processing_time/")

    # total_partial_num/partial_ratio/max_partial_span/param_version_diversity:
    # verl's fully_async equivalents, computed here from the per-sample
    # weight_versions history that gateway responses already attach (no new
    # instrumentation needed -- see Sample.oldest_weight_version/newest_weight_version).
    version_spans = [
        s.newest_weight_version - s.oldest_weight_version
        for s in all_samples
        if s.oldest_weight_version is not None and s.newest_weight_version is not None
    ]
    if version_spans:
        num_partial = sum(1 for span in version_spans if span > 0)
        metrics["total_partial_num"] = num_partial
        metrics["partial_ratio"] = num_partial / len(version_spans)
        metrics["max_partial_span"] = max(version_spans)

    all_versions_seen = {int(v) for s in all_samples for v in s.weight_versions if str(v).isdigit()}
    if all_versions_seen:
        metrics["param_version_diversity"] = len(all_versions_seen)

    # Duplicate of rollout/weight_version/mixed_version_ratio (see
    # _compute_metrics_from_samples), reported here too so every
    # partial-rollout-relevant metric can be viewed under one SwanLab group.
    # Deliberately not deduplicated: the original stays untouched.
    mixed = sum(1 for s in all_samples if len(set(s.weight_versions)) > 1)
    metrics["mixed_version_ratio"] = mixed / num_samples

    return metrics


def _compute_passrate_from_samples(args, all_samples: list[Sample]) -> dict[str, float]:
    """Compute pass@k metrics from samples using group_index for correct grouping.

    Unlike the trainer-side log_passrate (which assumed a flat reward array with
    contiguous groups of n_samples_per_prompt), this groups samples by their
    group_index field and computes pass@k over complete groups only. This is
    robust to filtering that may remove individual samples from a group —
    incomplete groups are excluded from the estimate rather than skewing it
    or crashing the reshape.

    Called on the rollout side (before convert_samples_to_train_data), so
    normally all samples are present and every group is complete.
    """
    group_size = args.n_samples_per_prompt
    if group_size <= 1:
        return {}

    # Fan-out trajectories are collapsed first (see _collapse_fanout_rewards):
    # per-segment grouping would make every fan-out group look "incomplete"
    # (K*n samples vs n) and silently drop pass@k entirely.
    pairs = _collapse_fanout_rewards(args, all_samples)
    groups = group_by(pairs, lambda pair: pair[0])
    completed_groups = [g for g in groups.values() if len(g) == group_size]
    if len(completed_groups) < len(groups):
        logger.warning(
            f"pass@k: excluding {len(groups) - len(completed_groups)}/{len(groups)} incomplete "
            f"groups (fewer than n_samples_per_prompt={group_size} samples)."
        )
    if not completed_groups:
        return {}

    flat_rewards = [reward for group in completed_groups for _, reward in group]

    return compute_pass_rate(
        flat_rewards=flat_rewards,
        group_size=group_size,
    )
