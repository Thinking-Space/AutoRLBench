#!/usr/bin/env python3
"""Extract the target veRL metric from fresh logs and assign a binary score."""

from __future__ import annotations

import argparse
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
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
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
    mode = value.get("score_mode", "best")
    if mode not in {"best", "final"}:
        raise ValueError("score_mode must be best or final")
    value["score_mode"] = mode
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
    expected_key = spec["metric_key"]
    step_line, eval_dict = metric_patterns(expected_key)
    values: dict[int, tuple[Any, Decimal, str]] = {}
    keys_seen: set[str] = {expected_key}
    source_logs = []
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
        matched = False
        for pattern in (step_line, eval_dict):
            for order, match in enumerate(pattern.finditer(text)):
                matched = True
                step = int(match.group(1))
                token = match.group(2)
                value = parse_number(token)
                if value is None:
                    continue
                rank = (mtime, relative, order)
                previous = values.get(step)
                if (
                    previous is None
                    or value > previous[1]
                    or (value == previous[1] and rank > previous[0])
                ):
                    values[step] = (rank, value, relative)
        if matched:
            source_logs.append(relative)

    points = []
    scores = []
    for step in sorted(values):
        score = float(values[step][1])
        points.append(
            {
                "step": step,
                "eval_score": score,
                "eval_key": expected_key,
                "truncate_ratio": None,
                "raw": {expected_key: score},
                "source_log": values[step][2],
            }
        )
        scores.append({"step": step, "score": score})
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": sorted(set(source_logs))},
        "metrics": {
            "metric_key": expected_key,
            "metric_status": "available" if points else "missing",
            "eval_points": points,
            "eval_scores": scores,
        },
    }
    return experiment, values, keys_seen


def select_best(values: dict[int, tuple[Any, Decimal, str]]):
    best = None
    for step, (rank, value, relative) in values.items():
        candidate = (value, step, rank, relative)
        if best is None or candidate > best:
            best = candidate
    return best


def judged_value(spec: dict[str, Any], values: dict[int, tuple[Any, Decimal, str]]):
    if spec["score_mode"] == "final":
        final_step = spec["final_step"]
        if final_step not in values:
            return None, None, None
        rank, value, relative = values[final_step]
        return value, final_step, relative
    best = select_best(values)
    if best is None:
        return None, None, None
    return best[0], best[1], best[3]


def grade(spec: dict[str, Any], results: Path):
    experiment, values, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    required_max = spec.get("required_final_score_max")
    required_max_value = (
        Decimal(str(required_max)) if required_max is not None else None
    )
    errors = []
    judged, judged_step, judged_source = judged_value(spec, values)
    improvement = judged - baseline if judged is not None else None
    if judged is None:
        if spec["score_mode"] == "final":
            errors.append(f"missing finite {metric_key} at step {final_step}")
        elif keys_seen:
            errors.append(f"missing finite {metric_key} in experiment_logs")
        else:
            errors.append(f"missing {metric_key} in experiment_logs")
    elif judged < required_score:
        errors.append(
            f"score {judged} at step {judged_step} is below "
            f"required score {required_score}"
        )
    elif required_max_value is not None and judged > required_max_value:
        errors.append(
            f"score {judged} at step {judged_step} is above "
            f"required maximum {required_max_value}"
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
        "best_step": judged_step,
        "best_score": judged_float,
        "best_source_log": judged_source,
        "final_score": judged_float,
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "source_logs": experiment["logs"]["source_logs"],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment["metrics"]["best_step"] = judged_step
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
