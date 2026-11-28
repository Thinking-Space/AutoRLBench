#!/usr/bin/env python3
"""Grade recovery from fresh VeRL per-batch metrics and checkpoint artifacts."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re


STEP = re.compile(r"\bstep:(\d+) - ")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SLEEP = re.compile(r"\[qat-mem\] sleep\(level=2\):[^\n]*\(freed ([\d.]+)\)")
FIELDS = (
    "training/global_step",
    "response_length/min",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "actor/lr",
    "actor/grad_norm",
)


def verify_workspace(workspace, spec):
    for path, expected in spec["protected_files"].items():
        if hashlib.sha256((workspace / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Protected launch or metric file changed: {path}")
    if not (workspace / "report.md").read_text(encoding="utf-8").strip():
        raise ValueError("report.md is empty")


def grade_run(logs, spec):
    final_step = spec["final_step"]
    checkpoints = logs / "checkpoints"
    if (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip() != str(final_step):
        raise ValueError(f"Expected native checkpoint marker {final_step}")
    checkpoint = checkpoints / f"global_step_{final_step}"
    required = [checkpoint / "data.pt"]
    required.extend(
        checkpoint / "actor" / f"{kind}_world_size_2_rank_{rank}.pt"
        for rank in range(2)
        for kind in ("model", "optim", "extra_state")
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError("Final checkpoint is missing a model, optimizer, RNG, or data file")

    batches = {}
    releases = 0
    batch_size = spec["responses_per_step"]
    limit = spec["output_limit"]
    with (logs / "train.stdout").open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = ANSI.sub("", raw)
            sleep = SLEEP.search(line)
            if sleep and Decimal(sleep[1]) > 0:
                releases += 1
            step = STEP.search(line)
            if not step:
                continue
            index = int(step[1])
            if index in batches:
                raise ValueError(f"Duplicate metrics for step {index}")
            values = {}
            for key in FIELDS:
                match = re.search(rf"(?:^| - ){re.escape(key)}:([^ \n]+)", line)
                if not match:
                    raise ValueError(f"Missing {key} at step {index}")
                value = Decimal(match[1])
                if not value.is_finite():
                    raise ValueError(f"Non-finite {key} at step {index}")
                values[key] = value
            capped = values["response_length/clip_ratio"] * batch_size
            low, mean, high = (values[f"response_length/{field}"] for field in ("min", "mean", "max"))
            if not 1 <= low <= mean <= high <= limit:
                raise ValueError(f"Impossible response lengths at step {index}")
            if not (0 <= capped <= batch_size and capped == capped.to_integral_value()):
                raise ValueError(f"Truncation metric does not represent {batch_size} responses at step {index}")
            if bool(capped) != (high == limit) or (capped == batch_size) != (low == limit):
                raise ValueError(f"Truncation count and lengths disagree at step {index}")
            if values["training/global_step"] != index or values["actor/lr"] != 0 or values["actor/grad_norm"] < 0:
                raise ValueError(f"Native optimization metrics disagree with the recipe at step {index}")
            batches[index] = {"step": index, "capped": int(capped), "mean_tokens": float(mean)}
    if list(batches) != list(range(1, final_step + 1)):
        raise ValueError(f"Expected training metrics for steps 1 through {final_step}")
    if releases < final_step:
        raise ValueError(f"Expected level-2 memory release across the rerun, saw {releases} records")
    score = min(1 - row["capped"] / batch_size for row in batches.values())
    return score, list(batches.values()), releases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / "resources/grader_spec.json").read_text())
    report = {"passed": False, "score": 0.0}
    code = 2
    try:
        workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
        verify_workspace(workspace, spec)
        score, batches, releases = grade_run(args.results / spec["log_roots"][0], spec)
        passed = spec["required_final_score"] <= score <= spec["required_final_score_max"]
        report.update(passed=passed, score=float(passed), observed_score=score, batches=batches, release_records=releases)
        if not passed:
            report["errors"] = [f"Lowest batch recovery {score:.4f}, required {spec['required_final_score']:.2f}"]
        code = 0 if passed else 1
    except (OSError, ValueError, InvalidOperation) as error:
        report["errors"] = [str(error)]
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
