#!/usr/bin/env python3
"""Bounded GPU1 probe using explicitly staged complete Megatron/Miles source."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.checkpoint_policy import check_guard
from runtime.framework_config import PROBE_SCOPE
from runtime.mtp_training import train_probe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, default=ROOT / "vendor/megatron")
    parser.add_argument("--miles-root", type=Path, default=ROOT / "vendor/miles")
    parser.add_argument("--hook-file", type=Path, default=ROOT / "runtime/source/weight_sync_hook.py")
    parser.add_argument("--guard-file", type=Path, default=ROOT / "runtime/source/checkpoint_args.sh")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_logs"))
    parser.add_argument("--steps", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--timeout-seconds", type=int, choices=range(1, 601), default=300)
    parser.add_argument("--guard-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.is_relative_to((ROOT / "logs").resolve()):
        raise ValueError("Output must be separate from historical workspace/logs")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    report = {"schema_version": 1, "evidence_kind": "new_reduced_framework_probe",
              "scope": PROBE_SCOPE, "seed": args.seed, "requested_steps": args.steps,
              "optimizer_steps": 0, "phase": "checkpoint_guard"}
    report["guard"] = check_guard(args.guard_file)
    # Probe the model independently: the guard must not conceal the MTP fault.
    # This deliberate bypass is new probe behavior, not the old launcher.
    report["guard_bypassed_for_model_probe"] = not args.guard_only
    def timed_out(*_):
        raise TimeoutError("Fixed smoke budget exhausted")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout_seconds)
    code = 0
    try:
        if not args.guard_only:
            train_probe(args, report)
    except Exception as exc:
        report["exception_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["stack"] = [{"file": Path(f.filename).name, "function": f.name,
                            "line": f.lineno} for f in traceback.extract_tb(exc.__traceback__)]
        code = 1
    finally:
        signal.alarm(0)
        (output / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evidence_kind": report["evidence_kind"], "phase": report["phase"],
                      "optimizer_steps": report["optimizer_steps"],
                      "exception_type": report.get("exception_type"),
                      "guard_accepted": report["guard"]["accepted"]}))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
