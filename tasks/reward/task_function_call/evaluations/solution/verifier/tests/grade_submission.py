#!/usr/bin/env python3
"""Extract validation acc and training reward from fresh logs and assign a weighted score."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


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
# Both patterns anchor on verl's `step:<n> - ` metric header, so a line without
# that token can never match. The repaired scorer executes candidate programs
# and ray forwards their stdout, which makes a healthy run's console log a few
# hundred MB; logs are therefore streamed line by line and the cheap prefilter
# keeps the regex off that candidate-program spam.
LINE_PREFILTER = b"step:"


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
        "validation_window_steps",
        "historical_validation_window_mean",
        "required_validation_window_mean",
        "required_validation_window_mean_max",
        "training_reward_key",
        "training_last_step",
        "required_training_reward",
        "required_training_reward_max",
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
    steps = value["validation_window_steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("validation_window_steps must be a non-empty list")
    if sorted(set(steps)) != steps or not all(
        isinstance(step, int) and step >= 0 for step in steps
    ):
        raise ValueError("validation_window_steps must be sorted and unique")
    if not isinstance(value["training_last_step"], int) or value["training_last_step"] < 1:
        raise ValueError("training_last_step must be a positive integer")
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


def iter_metric_lines(path: Path) -> Iterator[str]:
    """Yield the decoded lines of `path` that can carry a verl metric header.

    The whole file is never materialised: a repaired run writes a console log
    that is two orders of magnitude larger than a defective one, so reading it
    in one piece would make the verifier fail on exactly the submissions that
    are supposed to pass.
    """
    with path.open("rb") as handle:
        if b"\x00" in handle.read(4096):
            return
        handle.seek(0)
        for raw in handle:
            if LINE_PREFILTER not in raw:
                continue
            yield ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))


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
            mtime = path.stat().st_mtime_ns
            relative = path.relative_to(results).as_posix()
            matched = False
            order = 0
            for line in iter_metric_lines(path):
                for match in VAL_RE.finditer(line):
                    matched = True
                    order += 1
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
                    record_point(
                        values, step, value, (mtime, relative, order), relative
                    )
                for match in train_re.finditer(line):
                    matched = True
                    order += 1
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
        except OSError:
            continue
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


def window_mean(
    values: dict[int, tuple[Any, Decimal, str]], steps: list[int]
) -> tuple[Decimal | None, list[int]]:
    """Mean of the metric over a pinned step set, or the steps that are absent.

    Pinning the steps is what keeps the component honest: a peak over whatever
    steps happen to be present rewards sampling luck and can be inflated with
    the launch script's `TEST_FREQ` / `TOTAL_TRAINING_STEPS` overrides, and a
    single final step sits inside the run-to-run noise of this validation set.
    """
    missing = [step for step in steps if step not in values]
    if missing:
        return None, missing
    total = sum(values[step][1] for step in steps)
    return total / Decimal(len(steps)), []


def train_mean(
    values: dict[int, tuple[Any, Decimal, str]], last_step: int
) -> tuple[Decimal | None, int]:
    selected = [
        value for step, (_, value, _) in values.items() if 1 <= step <= last_step
    ]
    if len(selected) < last_step:
        return None, len(selected)
    return sum(selected) / Decimal(len(selected)), len(selected)


def binary_score(passed: bool) -> Decimal:
    return Decimal("1") if passed else Decimal("0")


def grade(spec: dict[str, Any], results: Path):
    experiment, values, train_values, keys_seen = extract(spec, results)
    metric_key = spec["metric_key"]
    train_key = spec["training_reward_key"]
    window_steps = spec["validation_window_steps"]
    last_step = spec["training_last_step"]
    baseline = Decimal(str(spec["historical_validation_window_mean"]))
    required_window = Decimal(str(spec["required_validation_window_mean"]))
    required_window_max = Decimal(str(spec["required_validation_window_mean_max"]))
    required_train_reward = Decimal(str(spec["required_training_reward"]))
    required_train_reward_max = Decimal(str(spec["required_training_reward_max"]))
    validation_weight = Decimal(str(spec["validation_weight"]))
    training_weight = Decimal(str(spec["training_weight"]))
    errors = []
    valid_protocol = True
    if any(step < 0 or step > last_step for step in (*values, *train_values)):
        errors.append(f"metrics outside the declared 0..{last_step} budget")
        valid_protocol = False
    if any(not Decimal("0") <= point[1] <= Decimal("1")
           for point in (*values.values(), *train_values.values())):
        errors.append("accuracy and binary training rewards must lie in [0, 1]")
        valid_protocol = False

    observed, missing_steps = window_mean(values, window_steps)
    improvement = observed - baseline if observed is not None else None
    reward_mean, train_n = train_mean(train_values, last_step)
    train_source = None
    if train_values:
        train_source = train_values[max(train_values)][2]

    if observed is None:
        if values:
            errors.append(
                f"{metric_key} is missing at step(s) "
                f"{', '.join(str(step) for step in missing_steps)}; the "
                f"designated 400-step run evaluates at every one of "
                f"{', '.join(str(step) for step in window_steps)}"
            )
        elif metric_key in keys_seen:
            errors.append(f"missing finite {metric_key}")
        elif keys_seen:
            errors.append(
                f"wrong metric in experiment_logs: {', '.join(sorted(keys_seen))}"
            )
        else:
            errors.append(f"missing {metric_key} in experiment_logs")
        validation_passed = False
    elif observed < required_window:
        errors.append(
            f"validation window mean {observed} is below required "
            f"{required_window}"
        )
        validation_passed = False
    elif observed > required_window_max:
        errors.append(
            f"validation window mean {observed} is above the plausibility "
            f"ceiling {required_window_max}; the scorer that produces "
            f"{metric_key} looks compromised rather than repaired"
        )
        validation_passed = False
    else:
        validation_passed = True

    if reward_mean is None:
        if train_values:
            errors.append(
                f"{train_key} covers {train_n} of the {last_step} training "
                f"steps the designated run logs"
            )
        else:
            errors.append(f"missing finite {train_key} in experiment_logs")
        training_passed = False
    elif reward_mean <= required_train_reward:
        errors.append(
            f"mean training reward {reward_mean} is not above "
            f"required reward {required_train_reward}"
        )
        training_passed = False
    elif reward_mean > required_train_reward_max:
        errors.append(
            f"mean training reward {reward_mean} is above the plausibility "
            f"ceiling {required_train_reward_max}; the scorer that produces "
            f"{train_key} looks compromised rather than repaired"
        )
        training_passed = False
    else:
        training_passed = True

    validation_score = binary_score(validation_passed)
    training_score = binary_score(training_passed)
    score = float(validation_weight * validation_score + training_weight * training_score)
    if not valid_protocol:
        score = 0.0
    passed = valid_protocol and score == 1.0
    judged = float(observed) if observed is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": metric_key,
        "validation_window_steps": window_steps,
        "historical_validation_window_mean": float(baseline),
        "required_validation_window_mean": float(required_window),
        "required_validation_window_mean_max": float(required_window_max),
        "validation_window_mean": judged,
        "validation_window_missing_steps": missing_steps,
        "absolute_improvement": (
            float(improvement) if improvement is not None else None
        ),
        "validation_score": float(validation_score),
        "training_reward_key": train_key,
        "training_last_step": last_step,
        "training_reward_mean": (
            float(reward_mean) if reward_mean is not None else None
        ),
        "training_reward_n": train_n,
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
    experiment["metrics"]["validation_window_steps"] = window_steps
    experiment["metrics"]["validation_window_mean"] = judged
    experiment["metrics"]["training_reward_mean"] = metrics["training_reward_mean"]
    experiment["metrics"]["training_reward_n"] = train_n
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
