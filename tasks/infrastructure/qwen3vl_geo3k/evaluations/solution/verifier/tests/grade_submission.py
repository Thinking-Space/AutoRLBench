#!/usr/bin/env python3
"""Grade a repaired workspace with fresh GPU measurements and its foreground run."""
import json
import os
from pathlib import Path
import subprocess
import sys

from metrics import repair_verdict
from parity import read, write, digest, SPEC_PATH
from training_evidence import inspect_training


def main():
    workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")).resolve()
    results = Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")).resolve()
    verifier = Path(__file__).resolve().parents[1]
    oracle = Path(os.environ.get("AUTORLBENCH_ORACLE", verifier.parent / "oracle"))
    spec = read(SPEC_PATH)
    try:
        qualification = read(oracle / "historical_evidence_manifest.json")
        if not qualification.get("release_ready", False):
            raise RuntimeError("Task construction has not passed the paired 8B GPU/RL qualification")
        asset_manifest_path = workspace / "data/asset_manifest.json"
        if digest(asset_manifest_path) != qualification["asset_manifest_sha256"]:
            raise RuntimeError("Prepared asset manifest changed")
        for name, expected in read(asset_manifest_path)["files"].items():
            path = workspace / name
            # Capture independently hashes all four immutable weight shards.
            if path.suffix == ".safetensors":
                continue
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise RuntimeError(f"Fixed model/data asset changed: {name}")
        fixtures = qualification["fixture_pins"]
        for name, expected_hash in fixtures.items():
            if digest(workspace / "data/parity" / name) != expected_hash:
                raise RuntimeError("Fixed diagnostic input assets changed")
        training = inspect_training(results / "experiment_logs", spec["required_training_steps"])
        if not training["passed"]:
            result = {"passed": False, "score": 0.0, "metrics": {"training": training}}
            write(results / "result.json", result)
            return 1
        out = results / "verifier_measurements"
        out.mkdir(parents=True, exist_ok=True)
        script = str(verifier / "tests/parity.py")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "PYTHONPATH": str(workspace / "verl")}
        commands = [
            [sys.executable, script, "capture", "--workspace", str(workspace), "--out", str(out / "capture.json")],
            [sys.executable, "-m", "torch.distributed.run", "--standalone",
             "--nproc_per_node=" + os.environ.get("AUTORLBENCH_SCORE_GPUS", "1"), script, "score",
             "--workspace", str(workspace), "--capture", str(out / "capture.json"), "--out", str(out / "scores.json")],
        ]
        with (out / "execution.log").open("w") as log:
            for command in commands:
                subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        parity = repair_verdict(read(out / "scores.json"), spec)
        result = {"passed": parity["passed"], "score": parity["score"],
                  "metrics": {"parity": parity, "training": training}}
        write(results / "metrics.json", result["metrics"])
        write(results / "result.json", result)
        return 0 if result["passed"] else 1
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        write(results / "result.json", {"passed": False, "score": 0.0,
              "metrics": {"status": "unavailable", "error": str(exc)}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
