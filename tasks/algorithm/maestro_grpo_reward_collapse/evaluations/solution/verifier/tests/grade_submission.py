#!/usr/bin/env python3
"""Extract the target Maestro GRPO metric from fresh logs and assign a binary score.

The graded metric is `critic/score/mean` at step 200 of the 200-step staged run.
A bare log line proves nothing, so the value is anchored four ways; only the
shapes and cross-relations of a real run are accepted, never a value on its
own:

* strict single-line extraction: a reading counts only when
  `step:<n> - ... critic/score/mean:<v>` appears on one line of one text log
  under experiment_logs (the console format verl's aggregate logger emits;
  nothing is concatenated across lines), and the supplying file must show a
  complete run -- at least min_step_coverage of the training steps 1..200.
* corroboration at the final step from the same file: the run must end with
  finished tool trajectories (`verl_tool/is_traj_finished/mean` >= 0.9) and an
  unclipped response distribution (`response_length/clip_ratio` <= 0.05). The
  recorded collapse violates both, so a repair that only fixes the logged
  number, or a truncated run that never enters the tool loop, cannot pass.
* required_log_patterns: the staged configuration (training and held-out
  datasets, model, GRPO group size, the six tool-service launches, the
  historical action-stop-token wiring and log routing) must still be visible
  in the judged log, so a run that swaps the data, the model or the budget
  fails even when its metric clears the bar.
* live grading (the stage sets AUTORLBENCH_WORKSPACE / AUTORLBENCH_VERIFIER)
  sha256-pins the launch and logging chain against grader_spec.json, so
  editing the runner, the launcher or the console logger fails; the repair
  itself lives in the agent-loop reconstruction and is not pinned.

A replay of packaged evidence by hand runs without the stage environment and
skips the workspace checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
NUMBER = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|nan|inf(?:inity)?)"
)
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".out", ".txt", ""}
MAX_FILE_BYTES = 128 * 1024 * 1024
FIRST_STEP = 1
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "required_log_patterns",
    "corroboration_metrics",
    "min_step_coverage",
    "pinned_sha256",
}
OPTIONAL_SPEC_KEYS = {"required_final_score_max", "score_mode"}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"refusing to replace symlink: {path}")
    temporary = path.with_name(f"{path.name}.tmp-{secrets.token_hex(4)}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("grader_spec.json must be an object")
    keys = set(value)
    if not REQUIRED_SPEC_KEYS <= keys:
        raise ValueError("grader_spec.json is missing required fields")
    if keys - REQUIRED_SPEC_KEYS - OPTIONAL_SPEC_KEYS:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 2:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    if not isinstance(value["final_step"], int) or value["final_step"] < FIRST_STEP:
        raise ValueError("final_step must be an integer at or above 1")
    if not isinstance(value["pinned_sha256"], dict) or not value["pinned_sha256"]:
        raise ValueError("grader_spec.json must pin at least one workspace file")
    patterns = value["required_log_patterns"]
    if not isinstance(patterns, list) or not patterns or not all(
        isinstance(entry, dict)
        and set(entry) == {"pattern", "description"}
        and isinstance(entry["pattern"], str)
        and isinstance(entry["description"], str)
        for entry in patterns
    ):
        raise ValueError(
            "required_log_patterns must be non-empty objects with pattern and description"
        )
    for entry in patterns:
        try:
            re.compile(entry["pattern"])
        except re.error as exc:
            raise ValueError(f"invalid required log pattern: {exc}")
    corroboration = value["corroboration_metrics"]
    if not isinstance(corroboration, list) or not corroboration or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("key"), str)
        and (set(entry) == {"key", "min"} or set(entry) == {"key", "max"})
        for entry in corroboration
    ):
        raise ValueError(
            "corroboration_metrics must be non-empty objects with key and min or max"
        )
    for entry in corroboration:
        try:
            Decimal(str(entry.get("min", entry.get("max"))))
        except InvalidOperation:
            raise ValueError("corroboration_metrics bounds must be numbers")
    coverage = value["min_step_coverage"]
    if not isinstance(coverage, (int, float)) or not 0 < coverage <= 1:
        raise ValueError("min_step_coverage must be within (0, 1]")
    if value.get("score_mode", "final") != "final":
        raise ValueError("only final-step scoring is supported")
    value["score_mode"] = "final"
    return value


def log_files(results: Path, roots: list[str]) -> list[Path]:
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() == ".wandb":
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def metric_patterns(metric_key: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    key = re.escape(metric_key)
    number = NUMBER
    step_line = re.compile(
        rf"(?:^|\s)step:(\d+)[ \t]+-[^\r\n]*?\b{key}:({number})(?![\w./])",
        re.IGNORECASE,
    )
    eval_dict = re.compile(
        rf"""eval\s+(\d+):\s*\{{[^{{}}]*['"]{key}['"]\s*:\s*({number})""",
        re.IGNORECASE | re.DOTALL,
    )
    return step_line, eval_dict


def parse_number(token: str) -> Decimal | None:
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return value


def extract(spec: dict[str, Any], results: Path):
    """One pass over the text logs: per file, the step readings for the graded
    metric and for every corroboration metric (single-line console format),
    plus which required configuration patterns the file shows."""
    expected_key = spec["metric_key"]
    corr_keys = [entry["key"] for entry in spec["corroboration_metrics"]]
    pattern_specs = spec["required_log_patterns"]
    records: dict[str, dict[str, Any]] = {}
    merged: dict[int, tuple[Any, Decimal, str]] = {}
    keys_seen: set[str] = set()
    for path in log_files(results, spec["log_roots"]):
        try:
            data = path.read_bytes()
            if b"\x00" in data[:4096]:
                continue
            text = ANSI_RE.sub("", data.decode("utf-8", errors="replace"))
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        relative = path.relative_to(results).as_posix()
        record = {
            "mtime": mtime,
            "metric_steps": {},
            "corroboration_steps": {key: {} for key in corr_keys},
            "patterns": [
                bool(re.search(entry["pattern"], text)) for entry in pattern_specs
            ],
            "matched": False,
        }
        for key in [expected_key, *corr_keys]:
            step_line, eval_dict = metric_patterns(key)
            bucket = (
                record["metric_steps"]
                if key == expected_key
                else record["corroboration_steps"][key]
            )
            for pattern in (step_line, eval_dict):
                for match in pattern.finditer(text):
                    record["matched"] = True
                    keys_seen.add(key)
                    step = int(match.group(1))
                    value = parse_number(match.group(2))
                    if value is None:
                        continue
                    previous = bucket.get(step)
                    if previous is None or value > previous:
                        bucket[step] = value
        records[relative] = record
        rank = (mtime, relative)
        for step, value in record["metric_steps"].items():
            previous = merged.get(step)
            if (
                previous is None
                or value > previous[1]
                or (value == previous[1] and rank > previous[0])
            ):
                merged[step] = (rank, value, relative)

    points = []
    scores = []
    for step in sorted(merged):
        score = float(merged[step][1])
        points.append(
            {
                "step": step,
                "eval_score": score,
                "eval_key": expected_key,
                "truncate_ratio": None,
                "raw": {expected_key: score},
                "source_log": merged[step][2],
            }
        )
        scores.append({"step": step, "score": score})
    experiment = {
        "task_id": spec["task_id"],
        "logs": {
            "source_logs": sorted(
                relative for relative, record in records.items() if record["matched"]
            )
        },
        "metrics": {
            "metric_key": expected_key,
            "metric_status": "available" if points else "missing",
            "eval_points": points,
            "eval_scores": scores,
        },
    }
    return experiment, records, keys_seen


def run_continuity(
    spec: dict[str, Any], records: dict[str, dict[str, Any]]
) -> tuple[Decimal | None, str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Pick the log the run is judged from: it must supply the metric at the
    final step, show at least min_step_coverage of the training steps
    FIRST_STEP..final_step, and match every required configuration pattern.
    Among the files that qualify, the highest final-step reading wins, so the
    verdict never depends on which eligible log is newest."""
    final_step = spec["final_step"]
    expected = list(range(FIRST_STEP, final_step + 1))
    eligible = []
    rejected = []
    for relative, record in records.items():
        if not record["matched"]:
            continue
        coverage = (
            sum(1 for step in expected if step in record["metric_steps"])
            / len(expected)
        )
        missing = [
            entry["description"]
            for entry, ok in zip(spec["required_log_patterns"], record["patterns"])
            if not ok
        ]
        details = {
            "source_log": relative,
            "step_coverage": coverage,
            "missing_patterns": missing,
        }
        if (
            final_step in record["metric_steps"]
            and coverage >= spec["min_step_coverage"]
            and not missing
        ):
            eligible.append(
                (record["metric_steps"][final_step], record["mtime"], relative, details)
            )
        else:
            rejected.append(details)
    if not eligible:
        return None, None, None, rejected
    value, _, relative, details = max(eligible)
    return value, relative, details, rejected


def corroboration_errors(
    spec: dict[str, Any], record: dict[str, Any], relative: str
) -> tuple[list[str], dict[str, Any]]:
    final_step = spec["final_step"]
    errors = []
    values: dict[str, Any] = {}
    for entry in spec["corroboration_metrics"]:
        key = entry["key"]
        value = record["corroboration_steps"].get(key, {}).get(final_step)
        values[key] = float(value) if value is not None else None
        if value is None:
            errors.append(
                f"{relative} step {final_step}: missing corroboration metric "
                f"{key} (a real run logs it on the same console line)"
            )
            continue
        if "min" in entry and value < Decimal(str(entry["min"])):
            errors.append(
                f"{relative} step {final_step}: {key} is {value}, below "
                f"{entry['min']} (the recorded collapse ends with broken-off "
                "tool trajectories)"
            )
        if "max" in entry and value > Decimal(str(entry["max"])):
            errors.append(
                f"{relative} step {final_step}: {key} is {value}, above "
                f"{entry['max']} (the recorded collapse ends with clipped "
                "responses)"
            )
    return errors, values


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace, so live grading
    sha256-pins the files a legitimate repair does not touch: the runner and
    the launch/logging chain (budget contract, launch flags, console logger).
    A replay of packaged evidence by hand runs without the stage environment
    and skips the check."""
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    details: dict[str, Any] = {}
    for rel, expected in spec.get("pinned_sha256", {}).items():
        try:
            digest = hashlib.sha256((Path(workspace) / rel).read_bytes()).hexdigest()
        except OSError:
            digest = ""
        details[rel] = digest == expected
    return (all(details.values()) if details else None), details


def grade(spec: dict[str, Any], results: Path):
    experiment, records, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    required_max = spec.get("required_final_score_max")
    required_max_value = (
        Decimal(str(required_max)) if required_max is not None else None
    )
    errors = []
    corr_values: dict[str, Any] | None = None
    judged, judged_source, continuity, rejected = run_continuity(spec, records)
    improvement = judged - baseline if judged is not None else None
    if judged is None:
        if not any(record["matched"] for record in records.values()):
            errors.append(f"missing {metric_key} in experiment_logs")
        elif not any(
            final_step in record["metric_steps"]
            for record in records.values()
            if record["matched"]
        ):
            errors.append(f"missing finite {metric_key} at step {final_step}")
        elif rejected:
            best = max(rejected, key=lambda details: details["step_coverage"])
            if best["step_coverage"] < spec["min_step_coverage"]:
                errors.append(
                    f"{best['source_log']} shows "
                    f"{best['step_coverage']:.1%} of training steps "
                    f"{FIRST_STEP}..{final_step}; a complete run logs at least "
                    f"{spec['min_step_coverage']:.1%}, so a single fabricated or "
                    "truncated log line is not a run"
                )
            if best["missing_patterns"]:
                errors.append(
                    f"{best['source_log']} is missing the staged configuration "
                    f"anchors ({'; '.join(best['missing_patterns'])}); the run "
                    "must keep the staged datasets, model, group size, tool "
                    "services and launcher wiring"
                )
            if not best["missing_patterns"] and (
                best["step_coverage"] >= spec["min_step_coverage"]
            ):
                errors.append(f"missing finite {metric_key} at step {final_step}")
        else:
            errors.append(f"missing finite {metric_key} at step {final_step}")
    else:
        if judged < required_score:
            errors.append(
                f"score {judged} at step {final_step} is below "
                f"required score {required_score}"
            )
        elif required_max_value is not None and judged > required_max_value:
            errors.append(
                f"score {judged} at step {final_step} is above "
                f"required maximum {required_max_value}"
            )
        corr_errors, corr_values = corroboration_errors(
            spec, records[judged_source], judged_source
        )
        errors.extend(corr_errors)
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the workspace as shipped apart from the repair: "
            "the launch and logging chain is sha256-pinned at grading time, so "
            "editing the runner, the launcher or the console logger fails"
        )
    passed = not errors
    score = 1.0 if passed else 0.0
    judged_float = float(judged) if judged is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "final_step": final_step,
        "score_mode": spec["score_mode"],
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required_score),
        "required_final_score_max": (
            float(required_max_value) if required_max_value is not None else None
        ),
        "best_step": final_step if judged is not None else None,
        "best_score": judged_float,
        "best_source_log": judged_source,
        "final_score": judged_float,
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "source_logs": experiment["logs"]["source_logs"],
        "run_continuity": continuity if continuity is not None else None,
        "rejected_logs": rejected,
        "corroboration": corr_values,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": integrity_details,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment["metrics"]["best_step"] = final_step if judged is not None else None
    experiment["metrics"]["best_score"] = judged_float
    experiment["metrics"]["best_source_log"] = judged_source
    return experiment, metrics, {"passed": passed, "score": score, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")),
    )
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    try:
        spec = load_spec(
            Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json"
        )
        experiment, metrics, result = grade(spec, results)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metrics = {
            "score": 0.0,
            "case_score": 0.0,
            "passed": False,
            "errors": [f"verifier error: {exc}"],
            "deterministic": True,
            "judge_used": False,
        }
        result = {"passed": False, "score": 0.0, "metrics": metrics}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", result)
        except OSError:
            pass
        print(metrics["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
