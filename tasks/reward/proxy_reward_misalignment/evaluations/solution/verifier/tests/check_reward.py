"""Deterministic behavioral checks, separate from submitted scalar metrics.

The submitted module runs in a child process; expected scores stay in the
parent. This is process separation, not an OS security boundary. The harness
must isolate the verifier from untrusted training code for hostile submissions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def cases():
    rows = []
    for source in ("autorlbench_type3_proxy_train_v2", "math__math", "math__aime_repeated_8x"):
        for truth, wrong in [("23", "24"), (r"\frac{17}{8}", r"\frac{19}{8}"),
                             (r"\sqrt{7}", r"\sqrt{11}")]:
            for prefix in ("", "Reasoning. " * 500):
                for answer, expected in [(truth, 1.0), (wrong, 0.0)]:
                    rows.append(({"data_source": source, "solution_str": prefix + r"\boxed{" + answer + "}",
                                  "ground_truth": truth}, expected))
        for output, expected in [("", 0.0), ("I cannot solve this.", 0.0),
                                 (r"\boxed{24} then \boxed{23}", 1.0),
                                 (r"\boxed{23} then \boxed{24}", 0.0)]:
            rows.append(({"data_source": source, "solution_str": output, "ground_truth": "23"}, expected))
    return rows


def validate_outputs(outputs):
    fixtures = cases()
    if not isinstance(outputs, list) or len(outputs) != len(fixtures):
        raise ValueError("reward worker returned the wrong number of cases")
    errors = []
    for index, (actual, (_, expected)) in enumerate(zip(outputs, fixtures)):
        try:
            score = float(actual["score"] if isinstance(actual, dict) else actual)
            if not math.isfinite(score) or abs(score-expected) > 1e-8:
                errors.append(f"case {index}: expected {expected}, got {score}")
            if isinstance(actual, dict) and "acc" in actual and float(actual["acc"]) != expected:
                errors.append(f"case {index}: acc disagrees with correctness")
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"case {index}: invalid reward result")
    return {"passed": not errors, "cases": len(fixtures), "errors": errors}


def verify_assets(workspace):
    spec = json.loads((RESOURCES / "protected_assets.json").read_text())
    for name, expected in spec["files"].items():
        path = workspace / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(workspace.resolve()):
            raise ValueError(f"protected asset missing or redirected: {name}")
        if path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"protected asset changed: {name}")
    return spec["dataset_version"]


def check(workspace):
    version = verify_assets(workspace)
    worker = Path(__file__).with_name("reward_worker.py")
    process = subprocess.run(
        [sys.executable, "-I", str(worker), str(workspace)],
        input=json.dumps([request for request, _ in cases()]), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, cwd=workspace,
    )
    if process.returncode:
        raise ValueError(f"reward worker failed ({process.returncode}): {process.stderr[-2000:]}")
    marker = "AUTORLBENCH_REWARD_OUTPUT="
    lines = [line[len(marker):] for line in process.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise ValueError("reward worker output missing or ambiguous")
    result = validate_outputs(json.loads(lines[0]))
    result["dataset_version"] = version
    return result
