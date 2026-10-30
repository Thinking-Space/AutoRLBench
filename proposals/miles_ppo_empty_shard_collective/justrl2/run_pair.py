#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run intact and mutated JustRL2 replay sequentially in isolated task/container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

from verify_run import verify_healthy, verify_faulty
from checkpoint_evidence import verify_conversion


def run_arm(workspace, output, env, timeout):
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **env, "BENCH_RUN_DIR": str(output), "RAY_USAGE_STATS_ENABLED": "0"}
    with (output / "stdout.log").open("w") as stream:
        proc = subprocess.Popen(["bash", "benchmark/train.sh"], cwd=workspace,
                                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            deadline = True
        finally:
            # The documented environment is a dedicated single-job container;
            # only this case owns its Ray runtime. A shared live Ray host is not supported.
            try:
                subprocess.run(["ray", "stop", "--force"], env=env, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=60, check=True)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
    return {"returncode": proc.returncode, "outer_deadline_reached": deadline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("BENCH_DEDICATED_CONTAINER") != "1":
        raise RuntimeError("Run inside a dedicated task/container with BENCH_DEDICATED_CONTAINER=1")
    prepared, output = args.prepared.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay = prepared / "replay/replay.pt"
    source_manifest = json.loads((prepared / "pair/pair_manifest.json").read_text())
    conversion = verify_conversion(prepared, args.native_model.resolve())
    replay_manifest = json.loads((prepared / "replay/manifest.json").read_text())
    if hashlib.sha256(replay.read_bytes()).hexdigest() != replay_manifest["files"]["replay.pt"]["sha256"]:
        raise RuntimeError("Replay differs from its preparation manifest")
    for kind in ["healthy", "faulty"]:
        workspace = prepared / "pair" / kind
        for name, expected_hash in source_manifest[kind + "_files"].items():
            if hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError(f"Prepared {kind} source changed: {name}")
    env = {"BENCH_MODEL": str(prepared / "model"), "BENCH_NATIVE_MODEL": str(args.native_model.resolve()),
           "BENCH_REPLAY": str(replay)}
    results = {"replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
               "source_revision": source_manifest["revision"],
               "native_checkpoint": conversion["native_checkpoint"], "arms": {}}
    for kind in ["healthy", "faulty"]:
        workspace = prepared / "pair" / kind
        record = run_arm(workspace, output / kind, env, args.timeout)
        results["arms"][kind] = record
        (output / "ATTEMPT.json").write_text(json.dumps(results, indent=2) + "\n")
        if kind == "healthy":
            if record["returncode"] != 0 or record["outer_deadline_reached"]:
                raise RuntimeError("Intact source did not finish; fault comparison is not valid")
            record.update(verify_healthy(output / kind))
        else:
            if record["returncode"] == 0:
                raise RuntimeError("Injected regression did not fail")
            record.update(verify_faulty(output / kind))
            # A captured, attributed NCCL failure is evidence; the outer timeout
            # alone is never enough. Retain whether forced cleanup was required.
        (output / "ATTEMPT.json").write_text(json.dumps(results, indent=2) + "\n")
    results["paired_training_verified"] = True
    (output / "PAIR_RESULT.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
