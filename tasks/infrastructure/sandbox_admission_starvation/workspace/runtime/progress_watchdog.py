"""Dependency-free progress watchdog for fully-async rollout."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def optional_positive_env_float(*names: str) -> float | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; watchdog disabled", name, raw)
            return None
        if value <= 0:
            logger.warning("Ignoring non-positive %s=%r; watchdog disabled", name, raw)
            return None
        return value
    return None


def check_accepted_progress_timeout(
    *,
    now: float,
    last_accepted_at: float,
    timeout_seconds: float | None,
    accepted: int,
    target: int,
    filtered_dropped: int,
) -> None:
    if timeout_seconds is None or now - last_accepted_at < timeout_seconds:
        return
    raise RuntimeError(
        "Fully-async rollout made no accepted-group progress for "
        f"{now - last_accepted_at:.1f}s (accepted={accepted}/{target}, "
        f"filtered_dropped={filtered_dropped}); refusing to sample forever"
    )
