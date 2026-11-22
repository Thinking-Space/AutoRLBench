#!/usr/bin/env python3
"""Re-score the fresh run's held-out GSM8K outputs with a fixed rule and assign a binary score.

The grader never reads a logged accuracy. It reads the per-item dump that the shipped
`eval_gsm8k.py` writes after training (`experiment_logs/eval_after.json.items.jsonl`, one row per
GSM8K test item in file order) and re-scores every row against its own copy of the gold answers
(`resources/gsm8k_test_gold.json`): a row is correct only if its output finished on its own
(`finish == "stop"`, not cut off at the token limit) and the number after its last `####` equals
the gold answer numerically. All 1,319 rows must be present.

Run-reality evidence: `metrics.jsonl` must hold one training row (with `loss`) for every
global step 1..final_step and the trainer's closing summary row, and `eval_after.log` must carry
the `eval <final_step>:` line. Their values are never scored.

Live grading (the stage sets AUTORLBENCH_WORKSPACE) also byte-compares the workspace runner
files listed in `pinned_runner` against the verifier's pristine copies and checks the data files
in `pinned_sha256`. The repair belongs in `workspace/trl`; the runner, the reward, the evaluation
and the data must be as shipped. A replay of packaged evidence by hand runs without the stage
environment and skips these workspace checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FINAL_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
MAX_FILE_BYTES = 256 * 1024 * 1024


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def to_number(token: str) -> Decimal | None:
    token = token.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_dump(path: Path, gold: list[Decimal | None]) -> tuple[int, int, list[str]]:
    rows = correct = 0
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {line_no}: not JSON")
                continue
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                problems.append(f"line {line_no}: no text field")
                continue
            index = rows
            rows += 1
            if index >= len(gold):
                continue
            if item.get("finish") != "stop":
                continue
            found = FINAL_RE.findall(item["text"])
            if found and gold[index] is not None and to_number(found[-1]) == gold[index]:
                correct += 1
    return rows, correct, problems


def training_reality(log_dir: Path, final_step: int) -> tuple[dict[str, Any], str | None]:
    metrics_path = log_dir / "metrics.jsonl"
    details: dict[str, Any] = {"metrics.jsonl": str(metrics_path.name)}
    if not regular_file(metrics_path):
        return details, "missing experiment_logs/metrics.jsonl (the run must train through train_grpo.py as shipped)"
    steps: set[int] = set()
    summary = False
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        step = row.get("global_step")
        if "loss" in row and isinstance(step, int):
            steps.add(step)
        if "train_runtime" in row and row.get("global_step") == final_step:
            summary = True
    missing = sorted(set(range(1, final_step + 1)) - steps)
    details.update({"training_steps_logged": len(steps), "missing_steps": missing[:20], "trainer_summary_row": summary})
    if missing or not summary:
        return details, (
            f"metrics.jsonl does not show a complete {final_step}-step run "
            f"({len(missing)} training steps missing, trainer summary row present: {summary})"
        )
    eval_log = log_dir / "eval_after.log"
    if not regular_file(eval_log) or f"eval {final_step}:" not in eval_log.read_text(encoding="utf-8", errors="replace"):
        return details, f"experiment_logs/eval_after.log has no 'eval {final_step}:' line"
    return details, None


def stage_integrity(spec: dict[str, Any], verifier: Path) -> tuple[bool | None, dict[str, bool]]:
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE")
    if not workspace:
        return None, {}
    root = Path(workspace)
    details: dict[str, bool] = {}
    for rel in spec["pinned_runner"]:
        mine, pristine = root / rel, verifier / "resources" / "pristine" / rel
        details[rel] = regular_file(mine) and mine.read_bytes() == pristine.read_bytes()
    for rel, digest in spec["pinned_sha256"].items():
        path = root / rel
        details[rel] = regular_file(path) and sha256(path) == digest
    return all(details.values()), details


def grade(spec: dict[str, Any], verifier: Path, results: Path) -> dict[str, Any]:
    final_step = int(spec["final_step"])
    required = Decimal(str(spec["required_final_score"]))
    baseline = Decimal(str(spec["historical_final_baseline"]))
    gold_raw = json.loads((verifier / "resources" / "gsm8k_test_gold.json").read_text(encoding="utf-8"))["gold"]
    gold = [to_number(str(g)) for g in gold_raw]
    log_dir = results / spec["log_root"]
    dump = log_dir / spec["dump"]
    errors: list[str] = []
    rows = correct = 0
    problems: list[str] = []
    accuracy = None
    if not regular_file(dump):
        errors.append(f"missing {spec['log_root']}/{spec['dump']} (written by eval_gsm8k.py after training)")
    else:
        rows, correct, problems = score_dump(dump, gold)
        if rows != int(spec["rows"]):
            errors.append(f"{spec['dump']} has {rows} rows; the full GSM8K test split has {spec['rows']}")
        else:
            accuracy = Decimal(correct) / Decimal(rows)
            if accuracy < required:
                errors.append(f"strict held-out accuracy {accuracy:.4f} at step {final_step} is below required {required}")
    reality, reality_error = training_reality(log_dir, final_step)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity = stage_integrity(spec, verifier)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the shipped runner, reward, evaluation and data: failed workspace checks "
            f"{sorted(k for k, ok in integrity.items() if not ok)}"
        )
    passed = not errors
    score = 1.0 if passed else 0.0
    return {
        "task_id": spec["task_id"],
        "metric_key": spec["metric_key"],
        "final_step": final_step,
        "score_mode": "final",
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required),
        "final_score": float(accuracy) if accuracy is not None else None,
        "absolute_improvement": float(accuracy - baseline) if accuracy is not None else None,
        "scored_rows": rows,
        "correct_rows": correct,
        "dump_problems": problems[:20],
        "training_reality": reality,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": integrity,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    verifier = Path(__file__).resolve().parent.parent
    try:
        spec = json.loads((verifier / "resources" / "grader_spec.json").read_text(encoding="utf-8"))
        metrics = grade(spec, verifier, results)
        result = {"passed": metrics["passed"], "score": metrics["score"], "metrics": metrics}
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        print(json.dumps({"passed": metrics["passed"], "final_score": metrics["final_score"]}))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        metrics = {"score": 0.0, "case_score": 0.0, "passed": False, "errors": [f"verifier error: {exc}"],
                   "deterministic": True, "judge_used": False}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", {"passed": False, "score": 0.0, "metrics": metrics})
        except OSError:
            pass
        print(metrics["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
