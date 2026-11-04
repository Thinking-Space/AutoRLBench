#!/usr/bin/env python3
"""Re-score the fresh run's held-out generation dump with a fixed strict GSM8K rule and assign a binary score.

The task's defect is the reward function, which verl also uses for the logged validation metric, so the logged
`val-core/openai/gsm8k/acc/mean@1` cannot be trusted: a run that keeps (or loosens) the forgiving rule logs a high
number while never committing to a final answer. The grader therefore ignores logged metric values and scores the
responses the run dumped at `final_step` (`trainer.validation_data_dir` = `experiment_logs/val_generations`):
a response is correct iff the last `#### <number>` it contains equals the gold answer numerically.

The rerun is anchored so the scored dump can only come from the shipped experiment: live grading (the stage sets
AUTORLBENCH_WORKSPACE / AUTORLBENCH_VERIFIER) byte-compares the workspace `run_gsm8k_grpo.sh` against the
verifier's pristine copy and checks the GSM8K parquets against pinned sha256s, and every grading additionally
requires run-reality evidence in `metrics.jsonl` (verl's file logger writes one JSON line per training step
0..final_step, and the final evaluation must be logged) -- never the metric values. A replay of packaged
evidence by hand runs without the stage environment and skips the workspace checks.
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

MAX_FILE_BYTES = 512 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "dump_subdir",
    "min_rows",
    "pinned_sha256",
}
FINAL_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
LAST_RE = re.compile(r"-?\d+(?:\.\d+)?")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_SPEC_KEYS - set(value)
    if missing:
        raise ValueError(f"grader_spec.json missing keys: {sorted(missing)}")
    if int(value["schema_version"]) != 3:
        raise ValueError("grader_spec.json schema_version must be 3")
    value["final_step"] = int(value["final_step"])
    value["min_rows"] = int(value["min_rows"])
    return value


def to_number(token: str) -> Decimal | None:
    token = token.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def gold_number(gt: Any) -> Decimal | None:
    if gt is None:
        return None
    if isinstance(gt, dict):
        gt = gt.get("ground_truth", gt)
    text = str(gt)
    direct = to_number(text)
    if direct is not None:
        return direct
    found = LAST_RE.findall(text.replace(",", ""))
    return to_number(found[-1]) if found else None


def strict_correct(response: str, gt: Any) -> bool:
    """Last `#### <number>` in the response equals the gold answer (numeric comparison, 1e-6 relative)."""
    gold = gold_number(gt)
    if gold is None or not isinstance(response, str):
        return False
    finals = FINAL_RE.findall(response)
    if not finals:
        return False
    value = to_number(finals[-1])
    if value is None:
        return False
    tol = Decimal("1e-6") * max(Decimal(1), abs(gold))
    return abs(value - gold) <= tol


def find_dumps(results: Path, roots: list[str], subdir: str, final_step: int) -> list[Path]:
    """`<root>/**/<subdir>/<final_step>.jsonl`, regular files only (no symlinks anywhere below the root), newest first."""
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob(f"{final_step}.jsonl"):
            if path.is_symlink() or not path.is_file() or path.parent.name != subdir:
                continue
            rel = path.relative_to(root)
            if any((root / Path(*rel.parts[: i + 1])).is_symlink() for i in range(len(rel.parts) - 1)):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)


def score_dump(path: Path) -> tuple[int, int, list[str]]:
    rows = correct = 0
    problems: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {lineno}: not JSON")
                continue
            if not isinstance(row, dict) or "output" not in row or "gts" not in row:
                problems.append(f"line {lineno}: missing output/gts")
                continue
            rows += 1
            correct += strict_correct(row["output"], row["gts"])
    return rows, correct, problems


def find_metrics(results: Path, roots: list[str]) -> Path | None:
    """verl's file-logger output: `<root>/metrics.jsonl`, regular files only, newest first."""
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        direct = root / "metrics.jsonl"
        if direct.is_file() and not direct.is_symlink():
            found.append(direct)
            continue
        for path in root.rglob("metrics.jsonl"):
            if path.is_symlink() or not path.is_file() or "wandb" in path.relative_to(root).parts:
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)[0] if found else None


def training_reality(results: Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    """The scored dump is written by the step-`final_step` evaluation, so a run that actually
    executed the shipped runner leaves verl's file log (`VERL_FILE_LOGGER_PATH`, set by the
    runner to `<log root>/metrics.jsonl`) with one JSON line per training step 0..final_step
    and a `val-core` row at `final_step`. Only that shape is checked; the values are never
    scored, because the task's defect is the function that produces them."""
    final_step = spec["final_step"]
    val_key = "val-core/openai/gsm8k/acc/mean@1"
    path = find_metrics(results, spec["log_roots"])
    details: dict[str, Any] = {
        "metrics_path": str(path.relative_to(results)) if path else None,
        "metrics_rows": 0,
        "steps_continuous": False,
        "final_eval_logged": False,
    }
    if path is None:
        return False, details, (
            f"missing metrics.jsonl under {spec['log_roots']} (the shipped runner sets VERL_FILE_LOGGER_PATH "
            "and verl writes one JSON line per training step; a run that did not execute it has no "
            f"step-{final_step} evaluation to dump)"
        )
    rows: list[dict[str, Any]] = []
    first_problem = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                first_problem = first_problem or f"line {lineno}: not JSON"
                continue
            if isinstance(row, dict):
                rows.append(row)
    if first_problem:
        return False, details, f"{path.relative_to(results)}: malformed training log ({first_problem})"
    steps = [row["step"] for row in rows if isinstance(row.get("step"), int)]
    details["metrics_rows"] = len(rows)
    details["steps_continuous"] = steps == list(range(final_step + 1))
    details["final_eval_logged"] = any(
        row.get("step") == final_step and isinstance(row.get("data"), dict) and val_key in row["data"]
        for row in rows
    )
    if not details["steps_continuous"]:
        return False, details, (
            f"{path.relative_to(results)} does not show a complete run: expected one row per training step "
            f"0..{final_step} (verl's file logger), got {len(steps)} rows"
        )
    if not details["final_eval_logged"]:
        return False, details, (
            f"{path.relative_to(results)} has no {val_key} row at step {final_step}: the evaluation that "
            "writes the scored dump never ran"
        )
    return True, details, None


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace copy of the runner on the agent's data,
    so live grading compares them against the verifier's pins: the runner byte-for-byte
    against the pristine copy in resources/, the GSM8K parquets by sha256. A replay of
    packaged evidence by hand runs without the stage environment and skips the check."""
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    details: dict[str, Any] = {}
    try:
        details["run_gsm8k_grpo.sh"] = (Path(workspace) / "run_gsm8k_grpo.sh").read_bytes() == (
            Path(verifier) / "resources" / "run_gsm8k_grpo.sh"
        ).read_bytes()
    except OSError:
        details["run_gsm8k_grpo.sh"] = False
    for rel, expected in spec.get("pinned_sha256", {}).items():
        try:
            digest = hashlib.sha256((Path(workspace) / rel).read_bytes()).hexdigest()
        except OSError:
            digest = ""
        details[rel] = digest == expected
    return all(details.values()), details


def grade(spec: dict[str, Any], results: Path):
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required = Decimal(str(spec["required_final_score"]))
    required_max = spec.get("required_final_score_max")
    required_max = Decimal(str(required_max)) if required_max is not None else None
    errors: list[str] = []
    dumps = find_dumps(results, spec["log_roots"], spec["dump_subdir"], final_step)
    judged = None
    source = None
    rows = correct = 0
    problems: list[str] = []
    if not dumps:
        errors.append(
            f"missing {spec['dump_subdir']}/{final_step}.jsonl under {spec['log_roots']} "
            "(the run must reach its final evaluation with trainer.validation_data_dir as shipped)"
        )
    else:
        source = dumps[0]
        rows, correct, problems = score_dump(source)
        if rows < spec["min_rows"]:
            errors.append(f"{source.relative_to(results)} has {rows} scored rows, fewer than the {spec['min_rows']} required (full GSM8K test split)")
        else:
            judged = Decimal(correct) / Decimal(rows)
            if judged < required:
                errors.append(f"strict held-out accuracy {judged:.4f} at step {final_step} is below required {required}")
            elif required_max is not None and judged > required_max:
                errors.append(f"strict held-out accuracy {judged:.4f} at step {final_step} is above the maximum {required_max}")
    _, reality_details, reality_error = training_reality(results, spec)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the workspace as shipped apart from the repair: run_gsm8k_grpo.sh is "
            "byte-compared against the verifier's pristine copy and the GSM8K parquets are pinned by "
            "sha256 at grading time, so editing the runner, shortening the run or swapping the data fails"
        )
    passed = not errors
    score = 1.0 if passed else 0.0
    judged_f = float(judged) if judged is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": spec["metric_key"],
        "final_step": final_step,
        "score_mode": "final",
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required),
        "required_final_score_max": float(required_max) if required_max is not None else None,
        "best_step": final_step if judged is not None else None,
        "best_score": judged_f,
        "best_source_log": str(source.relative_to(results)) if source else None,
        "final_score": judged_f,
        "absolute_improvement": float(judged - baseline) if judged is not None else None,
        "scored_rows": rows,
        "correct_rows": correct,
        "dump_problems": problems[:20],
        "training_reality": reality_details,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": integrity_details,
        "source_logs": [str(p.relative_to(results)) for p in dumps],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": metrics["source_logs"]},
        "metrics": {
            "metric_key": spec["metric_key"],
            "metric_status": "available" if judged is not None else "missing",
            "best_step": metrics["best_step"],
            "best_score": judged_f,
            "best_source_log": metrics["best_source_log"],
            "scored_rows": rows,
            "correct_rows": correct,
        },
    }
    return experiment, metrics, {"passed": passed, "score": score, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    try:
        spec = load_spec(Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json")
        experiment, metrics, result = grade(spec, results)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metrics = {"score": 0.0, "case_score": 0.0, "passed": False, "errors": [f"verifier error: {exc}"], "deterministic": True, "judge_used": False}
        result = {"passed": False, "score": 0.0, "metrics": metrics}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", result)
        except OSError:
            pass
        print(metrics["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
