"""Launch the inventory experiment using its local configuration."""

import json
import os
from pathlib import Path
import subprocess
import sys


def build_command(workspace: Path, extra: list[str]) -> list[str]:
    config = json.loads((workspace / "case/config.json").read_text())
    model = Path(os.environ.get("MODEL_PATH", str(workspace / "model"))).expanduser().resolve()
    seed = int(os.environ.get("SEED", "42"))
    steps = int(os.environ.get("STEPS", "20"))
    if steps < 1:
        raise ValueError("STEPS must be positive; use trainer.val_only=true for evaluation only")
    config.update({
        "data.train_files": str(workspace / "data/train.parquet"),
        "data.val_files": str(workspace / "data/val.parquet"),
        "data.seed": seed,
        "actor_rollout_ref.model.path": str(model),
        "custom_reward_function.path": str(workspace / "case/reward.py"),
        "custom_reward_function.name": "compute_score",
        "trainer.total_training_steps": steps,
        "trainer.default_local_dir": str(workspace / "outputs/checkpoints"),
        "trainer.rollout_data_dir": str(workspace / "outputs/rollouts"),
        "trainer.validation_data_dir": str(workspace / "outputs/validation"),
        "hydra.run.dir": str(workspace / "outputs/hydra"),
    })
    command = [sys.executable, "-m", "verl.trainer.main_ppo"]
    command.extend(f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in config.items())
    command.extend(extra)
    return command


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    outputs = workspace / "outputs"
    outputs.mkdir(exist_ok=True)
    command = build_command(workspace, sys.argv[1:])
    (outputs / "launch.json").write_text(json.dumps({"argv": command}, indent=2) + "\n")
    return subprocess.call(command, cwd=workspace)


if __name__ == "__main__":
    raise SystemExit(main())
