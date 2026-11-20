"""Common train, ordinary validation, and clean held-out metrics."""
from __future__ import annotations

import numpy as np

from .rollout import CLEAN_SPLIT, NORMAL_SPLIT


def training_metrics(reward_tensor, extra):
    scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
    if not len(scores) or len(extra.get("strict_success", [])) != len(scores):
        raise ValueError("Native inventory training batch is missing trajectory scores")
    return {
        "inventory/train_reward": float(scores.mean()),
        "inventory/train_strict_success": float(np.mean(extra["strict_success"])),
        "inventory/train_progress_reward": float(np.mean(extra["progress_reward"])),
        "inventory/train_receipt_reward": float(np.mean(extra["terminal_reward"])),
    }


def validation_metrics(sources, extra):
    scores = np.asarray(extra["strict_success"], dtype=float)
    if len(scores) != len(sources):
        raise ValueError("Native inventory validation lost strict success scores")
    conventional = sources == NORMAL_SPLIT
    held_out = sources == CLEAN_SPLIT
    if not conventional.any() or not held_out.any() or not (conventional | held_out).all():
        raise ValueError("Expected ordinary and fixed held-out validation trajectories")
    return {
        "inventory/ordinary_success": float(scores[conventional].mean()),
        "inventory/unseen_success": float(scores[held_out].mean()),
        "inventory/ordinary_samples": int(conventional.sum()),
        "inventory/unseen_samples": int(held_out.sum()),
    }
