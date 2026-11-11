from __future__ import annotations

from collections.abc import Callable

from .model import Sample


def mark_generator_zero(sample: Sample) -> None:
    """Convert a budget-exhausted trajectory into its behavioral label."""
    sample.metadata["behavior_zero"] = True
    sample.metadata["behavior_reason"] = "turn_budget_exhausted"
    sample.reward = 0.0


def record_binary(sample: Sample, value: float) -> None:
    sample.reward = float(value)
    sample.metadata["reward_record"] = {
        "mode": "binary",
        "value": float(value),
    }


def normalize_group(
    samples: list[Sample],
    score_submitted: Callable[[list[Sample]], list[float]],
) -> list[float]:
    eligible: list[Sample] = []
    for sample in samples:
        # BUG: the generator has already assigned a valid behavioral zero.
        # Existing rewards are treated as opaque and bypass normalization.
        if sample.reward is not None:
            continue
        eligible.append(sample)

    if eligible:
        for sample, value in zip(
            eligible, score_submitted(eligible), strict=True
        ):
            record_binary(sample, value)
    return [float(sample.reward or 0.0) for sample in samples]


def accept_valid_group(samples: list[Sample]) -> tuple[bool, str | None]:
    valid = [
        sample
        for sample in samples
        if not sample.remove_sample and not sample.metadata.get("env_error")
    ]
    if len(valid) < 6:
        return False, f"insufficient_valid_rewards_{len(valid)}_of_8"

    records = [
        sample.metadata.get("reward_record")
        for sample in valid
    ]
    if any(
        not isinstance(record, dict) or record.get("mode") != "binary"
        for record in records
    ):
        return False, "non_binary_reward_mode"

    values = {float(record["value"]) for record in records}
    if values == {0.0}:
        return False, "solve_none"
    if values == {1.0}:
        return False, "solve_all"
    return True, None
