#!/usr/bin/env python3
"""Grade the fresh AReaL run's final held-out evaluation metric and assign a binary score.

Adapted from tasks/configuration/autorlbench_verl_grpo_rollout_top_p_off_policy (same v3
grader family, runner_pinned=false because the defect lives in the launch script). This
task's experiment writes no generation dump the grader could re-score, so the graded value
is the `eval-rollout/reward` the fresh run logs at the end of its third epoch (the primary
held-out metric the frozen evidence also uses). The value is trustworthy without pinning
the metric-emitting code only because that code is pinned: the workspace's `areal/` source
tree, the `examples/` entry points and the lock files are checked by sha256, so a run that
inflated the metric by editing the framework or the reward logic fails stage integrity.

Run-reality evidence is likewise log-shaped: the shipped runner's StatsLogger prints one
`Train step N/87 done.` line per training step and one `eval-rollout/reward` table row per
epoch, so a complete run leaves exactly 87 step lines with the final `Train step 87/87
done.` followed by the third (graded) evaluation row. A replay of packaged evidence by
hand runs without the stage environment and skips the workspace checks.
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
    "min_eval_rows",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "pinned_sha256",
}
TRAIN_RE = re.compile(r"Train step (\d+)/(\d+) done")
EVAL_RE = re.compile(r"eval-rollout/reward\s*│\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


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
    value["min_eval_rows"] = int(value["min_eval_rows"])
    return value


def to_number(token: str) -> Decimal | None:
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def find_main_logs(results: Path, roots: list[str]) -> list[Path]:
    """`<root>/**/main.log`, regular files only (no symlinks anywhere below the root), newest first."""
    found: list[Path] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("main.log"):
            if path.is_symlink() or not path.is_file():
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


def parse_run(log: Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    """A complete run prints one `Train step N/<total> done.` line per training step
    (StatsLogger) and one `eval-rollout/reward` table row at the end of each epoch; the
    final evaluation is the last row and must follow the final step line. Only this shape
    is checked for the step lines; the graded value is the last evaluation row."""
    final_step = spec["final_step"]
    text = log.read_text(encoding="utf-8", errors="replace")
    steps: list[tuple[int, int]] = [(int(a), int(b)) for a, b in TRAIN_RE.findall(text)]
    evals = [(m.start(), m.group(1)) for m in EVAL_RE.finditer(text)]
    details: dict[str, Any] = {
        "main_log": log.name,
        "train_step_lines": len(steps),
        "last_train_step": f"{steps[-1][0]}/{steps[-1][1]}" if steps else None,
        "steps_continuous": [s for s, _ in steps] == list(range(1, final_step + 1)),
        "total_matches_declared": bool(steps) and steps[-1][1] == final_step,
        "eval_rows": len(evals),
        "final_eval_after_final_step": bool(steps) and bool(evals) and evals[-1][0] > text.rfind(f"Train step {final_step}/{final_step} done"),
    }
    if not steps:
        return False, details, (
            f"{details['main_log']} has no `Train step` lines: the shipped runner's StatsLogger "
            "prints one per training step, so this run did not execute it"
        )
    if not details["steps_continuous"] or not details["total_matches_declared"]:
        return False, details, (
            f"{details['main_log']} does not show a complete run: expected one `Train step N/{final_step} done.` "
            f"line for every N in 1..{final_step}, got {len(steps)} lines ending at {details['last_train_step']}"
        )
    if len(evals) < spec["min_eval_rows"]:
        return False, details, (
            f"{details['main_log']} logs {len(evals)} `eval-rollout/reward` rows, fewer than the "
            f"{spec['min_eval_rows']} epoch-end evaluations the full budget produces"
        )
    if not details["final_eval_after_final_step"]:
        return False, details, (
            f"{details['main_log']} last `eval-rollout/reward` row does not follow the final "
            f"`Train step {final_step}/{final_step} done.` line"
        )
    return True, details, None


def training_reality(results: Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None, Decimal | None, list[Decimal]]:
    logs = find_main_logs(results, spec["log_roots"])
    if not logs:
        return False, {"main_log": None, "train_step_lines": 0, "eval_rows": 0}, (
            f"missing main.log under {spec['log_roots']} (the shipped runner writes one under its "
            "experiment log root; a run that did not execute it has no held-out evaluation to grade)"
        ), None, []
    log = logs[0]
    ok, details, error = parse_run(log, spec)
    try:
        details["main_log"] = str(log.relative_to(results))
    except ValueError:
        pass
    if error:
        return False, details, error, None, []
    text = log.read_text(encoding="utf-8", errors="replace")
    values = [to_number(v) for v in EVAL_RE.findall(text)]
    if not values or any(v is None for v in values):
        return False, details, f"{details['main_log']}: a logged `eval-rollout/reward` value is missing or not a finite number", None, []
    return True, details, None, values[-1], values


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace copy of the launch script on the staged
    data, model and framework, so live grading checks those against the verifier's pins. A
    task whose defect lives in the launch script sets "runner_pinned": false; the budget is
    then held by the log-shape checks and the pins, as in configuration/verl_grpo_rollout_top_p_off_policy.
    A replay of packaged evidence by hand runs without the stage environment and skips the check."""
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    details: dict[str, Any] = {}
    if spec.get("runner_pinned", True):
        try:
            details["run.sh"] = (Path(workspace) / "run.sh").read_bytes() == (
                Path(verifier) / "resources" / "run.sh"
            ).read_bytes()
        except OSError:
            details["run.sh"] = False
    for rel, expected in spec.get("pinned_sha256", {}).items():
        try:
            digest = hashlib.sha256((Path(workspace) / rel).read_bytes()).hexdigest()
        except OSError:
            digest = ""
        accepted = expected if isinstance(expected, list) else [expected]
        details[rel] = digest in accepted
    return all(details.values()), details


def grade(spec: dict[str, Any], results: Path):
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required = Decimal(str(spec["required_final_score"]))
    required_max = spec.get("required_final_score_max")
    required_max = Decimal(str(required_max)) if required_max is not None else None
    errors: list[str] = []
    ok, reality_details, reality_error, judged, all_values = training_reality(results, spec)
    if reality_error:
        errors.append(reality_error)
    if ok and judged is not None:
        if judged < required:
            errors.append(
                f"held-out evaluation {judged:.4f} at the end of epoch 3 (step {final_step}) is below required {required}"
            )
        elif required_max is not None and judged > required_max:
            errors.append(
                f"held-out evaluation {judged:.4f} at the end of epoch 3 (step {final_step}) is above the maximum {required_max}"
            )
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the workspace as shipped apart from the repair: failed workspace checks "
            f"{sorted(k for k, ok2 in integrity_details.items() if not ok2)[:12]} (and {max(0, sum(1 for ok2 in integrity_details.values() if not ok2) - 12)} more; "
            "files in pinned_sha256 by sha256)"
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
        "best_source_log": reality_details.get("main_log"),
        "final_score": judged_f,
        "absolute_improvement": float(judged - baseline) if judged is not None else None,
        "epoch_eval_values": [float(v) for v in all_values] if all_values else [],
        "training_reality": reality_details,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": {k: v for k, v in list(integrity_details.items())[:8]},
        "stage_integrity_failed_count": sum(1 for v in integrity_details.values() if not v),
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": [reality_details["main_log"]] if reality_details.get("main_log") else []},
        "metrics": {
            "metric_key": spec["metric_key"],
            "metric_status": "available" if judged is not None else "missing",
            "best_step": metrics["best_step"],
            "best_score": judged_f,
            "best_source_log": metrics["best_source_log"],
            "epoch_eval_values": metrics["epoch_eval_values"],
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
