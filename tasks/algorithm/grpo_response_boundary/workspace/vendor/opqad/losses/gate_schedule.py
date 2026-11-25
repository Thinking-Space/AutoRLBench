"""Validated piecewise recovery gates, indexed by one-based global updates."""

import math


def validate_gate_schedule(schedule, *, total_steps=None):
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("gate_schedule must be a nonempty list")
    expected_start = 1
    stages = []
    for stage in schedule:
        if not isinstance(stage, dict) or set(stage) != {"start_step", "end_step", "top_k", "rho"}:
            raise ValueError("gate_schedule stages require start_step, end_step, top_k and rho")
        start, end, top_k = (stage[key] for key in ("start_step", "end_step", "top_k"))
        if any(type(value) is not int for value in (start, end, top_k)):
            raise ValueError("gate_schedule steps and top_k must be integers")
        if start != expected_start or end < start or top_k <= 0:
            raise ValueError("gate_schedule must cover contiguous updates from 1 with positive top_k")
        rho = stage["rho"]
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 1:
            raise ValueError("gate_schedule rho must be finite and greater than 1")
        stages.append(dict(start_step=start, end_step=end, top_k=top_k, rho=float(rho)))
        expected_start = end + 1
    if total_steps is not None and stages[-1]["end_step"] != total_steps:
        raise ValueError("gate_schedule must end at total_training_steps")
    return stages


def gate_for_step(schedule, global_step):
    if type(global_step) is not int or global_step < 1:
        raise ValueError("gate_schedule requires the current one-based global_steps metadata")
    for stage in schedule:
        if stage["start_step"] <= global_step <= stage["end_step"]:
            return stage["top_k"], stage["rho"]
    raise ValueError(f"global update {global_step} is outside gate_schedule")
