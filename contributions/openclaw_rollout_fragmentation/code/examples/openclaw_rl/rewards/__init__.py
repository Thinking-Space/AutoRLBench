"""Selectable reward designs for Advisor-OpenClaw RL.

Each design lives in its own module so experiments can select a reward without
silently changing the semantics of an existing run.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

REWARD_MODES = {
    "static_penalty": "examples.openclaw_rl.rewards.static_penalty",
}


def get_reward_module(mode: str) -> ModuleType:
    try:
        module_path = REWARD_MODES[mode]
    except KeyError as exc:
        choices = ", ".join(sorted(REWARD_MODES))
        raise ValueError(f"unknown advisor reward mode {mode!r}; choose one of: {choices}") from exc
    return import_module(module_path)
