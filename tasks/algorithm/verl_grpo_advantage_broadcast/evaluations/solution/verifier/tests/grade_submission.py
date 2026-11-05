#!/usr/bin/env python3
"""Extract validation acc and training reward from fresh logs and assign a weighted score.

Validation is the mean of the pinned eval steps (all must be present); training
reward is the mean over steps 1..training_reward_steps (all must be present).
Both components have floors and ceilings."""

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
VAL_RE = re.compile(
    r"(?:^|\s)step:(\d+)\s+-.*?"
    r"\b(val-core/([^\s:]+)/(acc|reward)/mean@(\d+)):"
    rf"({NUMBER})",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".out", ".txt", ""}


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
    expected = {
        "schema_version",
        "task_id",
        "metric_key",
        "final_step",
        "eval_steps",
        "historical_final_baseline",
        "required_final_score",
        "required_final_score_max",
        "training_reward_key",
        "required_training_reward",
        "required_training_reward_max",
        "training_reward_steps",
        "validation_weight",
        "training_weight",
        "log_roots",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("grader_spec.json has unexpected fields")
    if value["schema_version"] != 3:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    eval_steps = value["eval_steps"]
    if (
        not isinstance(eval_steps, list)
        or not eval_steps
        or not all(isinstance(step, int) and step > 0 for step in eval_steps)
        or eval_steps != sorted(set(eval_steps))
        or eval_steps[-1] > value["final_step"]
    ):
        raise ValueError("eval_steps must be increasing unique steps <= final_step")
    reward_steps = value["training_reward_steps"]
    if not isinstance(reward_steps, int) or reward_steps <= 0:
        raise ValueError("training_reward_steps must be a positive integer")
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
            found.append(path)
    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def record_point(
    values: dict[int, tuple[Any, Decimal, str]],
    step: int,
    value: Decimal,
    rank: tuple[Any, ...],
    relative: str,
) -> None:
    previous = values.get(step)
    if (
        previous is None
        or value > previous[1]
        or (value == previous[1] and rank > previous[0])
    ):
        values[step] = (rank, value, relative)


def train_reward_re(key: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:^|\s)step:(\d+)\s+-.*?\b{re.escape(key)}:({NUMBER})",
        re.IGNORECASE,
    )


def extract(spec: dict[str, Any], results: Path):
    expected_key = spec["metric_key"]
    train_key = spec["training_reward_key"]
    train_re = train_reward_re(train_key)
    values: dict[int, tuple[Any, Decimal, str]] = {}
    train_values: dict[int, tuple[Any, Decimal, str]] = {}
    keys_seen: set[str] = set()
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
        for order, match in enumerate(VAL_RE.finditer(text)):
            matched = True
            step = int(match.group(1))
            key = match.group(2)
            token = match.group(6)
            keys_seen.add(key)
            if key != expected_key:
                continue
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            record_point(values, step, value, (mtime, relative, order), relative)
        for order, match in enumerate(train_re.finditer(text)):
            matched = True
            step = int(match.group(1))
            token = match.group(2)
            keys_seen.add(train_key)
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            record_point(
                train_values, step, value, (mtime, relative, order), relative
            )
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
    train_points = []
    for step in sorted(train_values):
        reward = float(train_values[step][1])
        train_points.append(
            {
                "step": step,
                "train_reward": reward,
                "train_key": train_key,
                "source_log": train_values[step][2],
            }
        )
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": sorted(set(source_logs))},
        "metrics": {
            "metric_key": expected_key,
            "metric_status": "available" if points else "missing",
            "eval_points": points,
            "eval_scores": scores,
            "train_reward_key": train_key,
            "train_reward_status": "available" if train_points else "missing",
            "train_reward_points": train_points,
        },
    }
    return experiment, values, train_values, keys_seen


def pinned_mean(
    values: dict[int, tuple[Any, Decimal, str]], steps: list[int]
) -> tuple[Decimal | None, list[int]]:
    """Mean of the pinned eval steps; every step must be present."""
    missing = [step for step in steps if step not in values]
    if missing:
        return None, missing
    return (
        sum(values[step][1] for step in steps) / Decimal(len(steps)),
        missing,
    )


def mean_reward(
    train_values: dict[int, tuple[Any, Decimal, str]], n_steps: int
) -> tuple[Decimal | None, list[int], dict[int, tuple[Any, Decimal, str]]]:
    """Mean over training steps 1..n_steps; every step must be present.

    Steps beyond n_steps are ignored, so a longer run is graded the same as
    one that stopped at n_steps.
    """
    window = {
        step: item for step, item in train_values.items() if 1 <= step <= n_steps
    }
    missing = [step for step in range(1, n_steps + 1) if step not in window]
    if missing:
        return None, missing, window
    return (
        sum(item[1] for item in window.values()) / Decimal(len(window)),
        missing,
        window,
    )


def binary_score(passed: bool) -> Decimal:
    return Decimal("1") if passed else Decimal("0")


def grade(spec: dict[str, Any], results: Path):
    experiment, values, train_values, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    train_key = spec["training_reward_key"]
    final_step = spec["final_step"]
    eval_steps = spec["eval_steps"]
    reward_steps = spec["training_reward_steps"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required_score = Decimal(str(spec["required_final_score"]))
    required_score_max = Decimal(str(spec["required_final_score_max"]))
    required_train_reward = Decimal(str(spec["required_training_reward"]))
    required_train_reward_max = Decimal(str(spec["required_training_reward_max"]))
    validation_weight = Decimal(str(spec["validation_weight"]))
    training_weight = Decimal(str(spec["training_weight"]))
    errors = []
    eval_mean, eval_missing = pinned_mean(values, eval_steps)
    train_mean, train_missing, train_window = mean_reward(train_values, reward_steps)
    train_source = None
    if train_window:
        latest_step = max(train_window)
        train_source = train_window[latest_step][2]

    if not values:
        if metric_key in keys_seen:
            errors.append(f"missing finite {metric_key}")
        elif keys_seen:
            errors.append(
                f"wrong metric in experiment_logs: {', '.join(sorted(keys_seen))}"
            )
        else:
            errors.append(f"missing {metric_key} in experiment_logs")
        validation_passed = False
    elif eval_missing:
        errors.append(
            f"missing validation at pinned steps {', '.join(map(str, eval_missing))}"
        )
        validation_passed = False
    elif eval_mean < required_score:
        errors.append(
            f"mean {metric_key} over steps {eval_steps} is {eval_mean}, "
            f"below required score {required_score}"
        )
        validation_passed = False
    elif eval_mean > required_score_max:
        errors.append(
            f"mean {metric_key} over steps {eval_steps} is {eval_mean}, "
            f"above ceiling {required_score_max}"
        )
        validation_passed = False
    else:
        validation_passed = True

    if not train_values:
        errors.append(f"missing finite {train_key} in experiment_logs")
        training_passed = False
    elif train_missing:
        errors.append(
            f"training log covers {len(train_window)} of {reward_steps} steps "
            f"(first missing: {train_missing[0]})"
        )
        training_passed = False
    elif train_mean <= required_train_reward:
        errors.append(
            f"mean training reward {train_mean} is not above "
            f"required reward {required_train_reward}"
        )
        training_passed = False
    elif train_mean > required_train_reward_max:
        errors.append(
            f"mean training reward {train_mean} is above ceiling "
            f"{required_train_reward_max}"
        )
        training_passed = False
    else:
        training_passed = True

    validation_score = binary_score(validation_passed)
    training_score = binary_score(training_passed)
    score = float(validation_weight * validation_score + training_weight * training_score)
    passed = score == 1.0
    judged = float(eval_mean) if eval_mean is not None else None
    improvement = eval_mean - baseline if eval_mean is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "final_step": final_step,
        "eval_steps": eval_steps,
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required_score),
        "required_final_score_max": float(required_score_max),
        "eval_mean": judged,
        "final_score": judged,
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "validation_score": float(validation_score),
        "training_reward_key": train_key,
        "training_reward_mean": (
            float(train_mean) if train_mean is not None else None
        ),
        "training_reward_steps": reward_steps,
        "training_reward_n": len(train_window),
        "training_reward_source_log": train_source,
        "required_training_reward": float(required_train_reward),
        "required_training_reward_max": float(required_train_reward_max),
        "training_reward_score": float(training_score),
        "source_logs": experiment["logs"]["source_logs"],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment["metrics"]["eval_mean"] = judged
    experiment["metrics"]["eval_steps"] = eval_steps
    experiment["metrics"]["training_reward_mean"] = metrics["training_reward_mean"]
    experiment["metrics"]["training_reward_steps"] = reward_steps
    experiment["metrics"]["training_reward_n"] = metrics["training_reward_n"]
    experiment["metrics"]["validation_score"] = metrics["validation_score"]
    experiment["metrics"]["training_reward_score"] = metrics["training_reward_score"]
    experiment["metrics"]["score"] = score
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
        spec = load_spec(Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json")
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
