#!/usr/bin/env python3
"""Reproduce the MTP zero-loss and one-rollout checkpoint failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.checkpoint_policy import validate_checkpoint_schedule
from runtime.mtp_training import Batch, train_step


RESULTS = {
    "configuration": {
        "num_rollouts": 1,
        "save_interval": 1,
        "debug_train_only": True,
        "mtp_loss_scaling_factor": 0.0,
    },
    "checkpoint_guard": {},
    "training_step": {},
}


def main() -> int:
    try:
        validate_checkpoint_schedule(
            num_rollouts=1, save_interval=1, debug_train_only=True
        )
        RESULTS["checkpoint_guard"] = {"accepted": True}
    except ValueError as exc:
        RESULTS["checkpoint_guard"] = {
            "accepted": False,
            "error": str(exc),
        }

    try:
        RESULTS["training_step"] = train_step(
            Batch(policy_loss=0.25, auxiliary_raw_loss=0.75),
            mtp_loss_scaling_factor=0.0,
        )
    except (TypeError, ValueError) as exc:
        RESULTS["training_step"] = {
            "optimizer_steps": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output = json.dumps(RESULTS, indent=2, sort_keys=True)
    print(output)
    log_dir = Path("experiment_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.json").write_text(output + "\n", encoding="utf-8")

    if RESULTS["checkpoint_guard"].get("accepted") and RESULTS["training_step"].get("optimizer_steps") == 1:
        return 0
    print(
        "reproduced: one-rollout checkpoint validation and zero auxiliary-loss "
        "training path both fail before a useful optimizer step",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
