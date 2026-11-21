#!/usr/bin/env python3
"""Replay the two recorded phase commands using a fresh output directory."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SOURCE_PATHS = ("train_base.sh", "verl", "case", "inventory_tasks", "mechanism_tasks")
DATA_PATHS = ("data/train.parquet", "data/val.parquet")
MODEL_PATHS = (
    "model/config.json", "model/tokenizer.json", "model/tokenizer_config.json",
    "model/model.safetensors.index.json", "model/model-00001-of-00002.safetensors",
    "model/model-00002-of-00002.safetensors",
)


def phases(workdir: Path, results: Path) -> list[dict]:
    plan = json.loads((WORKSPACE / "launch_plan.json").read_text())
    if not isinstance(plan, list) or any(not isinstance(phase, dict) for phase in plan):
        raise SystemExit("launch_plan.json must contain a list of phases")
    if [phase.get("name") for phase in plan] != ["checkpoint", "resume"]:
        raise SystemExit("Expected a checkpoint phase followed by a resume phase")

    def expand(text: str) -> str:
        return text.replace("<workspace>", str(workdir)).replace("<results>", str(results))

    for phase in plan:
        args = phase.get("args")
        env = phase.get("env", {})
        if (not isinstance(args, list) or not all(isinstance(arg, str) and "=" in arg for arg in args)
                or not isinstance(env, dict) or not all(isinstance(value, str) for value in env.values())):
            raise SystemExit(f"Invalid args or env in launch plan phase {phase['name']}")
        phase["command"] = ["bash", "train_base.sh", *map(expand, args)]
        phase["env"] = {key: expand(value) for key, value in env.items()}
    return plan


def run_environment(workdir: Path, temp: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Every phase starts a local Ray cluster and writes its own transport sockets.
    env.pop("RAY_ADDRESS", None)
    for name in ("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP",
                 "SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"):
        env.pop(name, None)
    ipc = temp / "ipc"
    ipc.mkdir(parents=True)
    env.update(
        MODEL_PATH=str(workdir / "model"), STEPS="30", SEED="42", N_GPUS="1",
        PYTHONHASHSEED="42", TOKENIZERS_PARALLELISM="false", VLLM_USE_V1="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), RAY_TMPDIR=str(temp),
        VLLM_RPC_BASE_PATH=str(ipc),
        SWANLAB_MODE="local", SWANLAB_LOG_DIR=str(workdir / "outputs/swanlab"),
        TENSORBOARD_DIR=str(workdir / "outputs/tensorboard"),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(workdir), env.get("PYTHONPATH")) if part)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true", help="print commands and asset status without replaying")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results = (args.results_dir or Path(os.environ.get("AUTORLBENCH_RESULTS") or
                                       WORKSPACE / "replay_results")).resolve()
    workdir = results / "experiment_logs" / "run"
    if (WORKSPACE / "outputs") in (results, *results.parents):
        raise SystemExit("Set a results directory outside the recorded outputs directory.")
    plan = phases(workdir, results)
    missing_source = [name for name in SOURCE_PATHS if not (WORKSPACE / name).exists()]
    missing_assets = [name for name in DATA_PATHS + MODEL_PATHS if not (WORKSPACE / name).is_file()]

    if args.doctor:
        print(json.dumps({
            "ready": not (missing_source or missing_assets or workdir.exists()),
            "missing_source": missing_source,
            "missing_assets": missing_assets,
            "workspace": str(workdir),
            "existing_replay_directory": workdir.exists(),
            "phases": [{"name": phase["name"], "command": shlex.join(phase["command"])} for phase in plan],
        }, indent=2))
        if missing_source:
            raise SystemExit(1)
        return

    if missing_source or missing_assets:
        raise SystemExit(f"Replay files missing: {', '.join(missing_source + missing_assets)}. See run.sh --doctor.")
    if workdir.exists():
        raise SystemExit("Replay directory already exists. Choose a fresh --results-dir.")
    workdir.mkdir(parents=True)
    for name in (*SOURCE_PATHS, "data", "model"):
        source = WORKSPACE / name
        (workdir / name).symlink_to(source, target_is_directory=source.is_dir())

    for phase in plan:
        name = phase["name"]
        if name == "resume" and not (workdir / "outputs/checkpoints/global_step_10/data.pt").is_file():
            raise SystemExit("The checkpoint phase did not save step-10 data-loader state.")
        # Docker and Ray enforce a short enough local path for IPC socket names.
        with tempfile.TemporaryDirectory(prefix="ab-inventory-", dir="/tmp") as tmp:
            env = run_environment(workdir, Path(tmp))
            env.update(phase["env"])
            completed = subprocess.run(phase["command"], cwd=workdir, env=env, check=False)
        console = workdir / "outputs/train.log"
        if console.is_file():
            shutil.copy2(console, workdir / "outputs" / f"train-{name}.log")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    metrics = workdir / "outputs/metrics.jsonl"
    if not metrics.is_file():
        raise SystemExit("Replay finished without the native metric log.")
    with metrics.open() as handle:
        final_step = max((json.loads(line).get("step", -1) for line in handle if line.strip()), default=-1)
    if final_step != 30:
        raise SystemExit(f"Replay recorded step {final_step}; expected step 30.")


if __name__ == "__main__":
    main()
