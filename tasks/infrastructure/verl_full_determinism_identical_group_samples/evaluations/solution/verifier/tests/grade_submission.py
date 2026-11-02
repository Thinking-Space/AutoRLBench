#!/usr/bin/env python3
"""Extract the target veRL metric from fresh logs and assign a binary score.

Task-local addition: when grader_spec.json names a `validation_dump`, the final-step validation generations are
re-graded with the verifier's own copy of verl's GSM8K scorer (gsm8k_reference.py, not the workspace's) against the
verifier's protected answer key (`answer_key`, one entry per test problem). Every problem of the key must appear in
the dump exactly once, answers come from the key rather than the dump's own `gts`, and the logged metric must agree
with that re-grade. A workspace whose reward, scorer, dump writer or logged numbers were edited cannot pass on its
own data.

Two further task-local checks read the training rollout dumps (`rollout_dump`):
- group diversity: at every training step the dump must hold `rollout_groups` prompts with `rollout_n` responses
  each, and fewer than `max_identical_group_share` of those groups may consist of identical responses;
- reproducibility: the rerun repeats the same command for a few steps under `repeat_root`, and the rollout dumps of
  `repeat_steps` must be identical (the same multiset of rows) in both runs. Files under `repeat_root` are not read
  for the metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
from collections import Counter
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
OPTIONAL_SPEC_KEYS = {
    "required_final_score_max",
    "score_mode",
    "validation_dump",
    "validation_recheck_tolerance",
    "answer_key",
    "rollout_dump",
    "rollout_n",
    "rollout_groups",
    "max_identical_group_share",
    "repeat_root",
    "repeat_steps",
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_reference  # noqa: E402


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
    if "rollout_dump" in value:
        for key in ("rollout_n", "rollout_groups", "max_identical_group_share"):
            if key not in value:
                raise ValueError(f"rollout_dump requires {key}")
    if "repeat_root" in value:
        steps = value.get("repeat_steps")
        if "rollout_dump" not in value or not isinstance(steps, list) or not steps:
            raise ValueError("repeat_root requires rollout_dump and a non-empty repeat_steps")
        if Path(value["repeat_root"]).is_absolute() or ".." in Path(value["repeat_root"]).parts:
            raise ValueError("repeat_root must be a relative path inside the log root")
    return value


def log_files(results: Path, roots: list[str], exclude: list[Path] | None = None) -> list[Path]:
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if any(path.is_relative_to(skip) for skip in exclude or []):
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
    exclude = []
    if "repeat_root" in spec:
        exclude.append(results / spec["log_roots"][0] / spec["repeat_root"])
    for path in log_files(results, spec["log_roots"], exclude):
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


def user_message(decoded_input: str) -> str | None:
    """The user turn of a verl validation dump's decoded prompt ("system ... user <message> assistant")."""
    start = decoded_input.find("\nuser\n")
    end = decoded_input.rfind("\nassistant")
    if start < 0 or end <= start:
        return None
    return decoded_input[start + len("\nuser\n"):end]


def question_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def load_answer_key(spec: dict[str, Any]) -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / "resources" / spec["answer_key"]
    problems = json.loads(path.read_text(encoding="utf-8"))["problems"]
    answers = {p["question_sha256"]: str(p["answer"]) for p in problems}
    if len(answers) != len(problems):
        raise ValueError("answer key has duplicate questions")
    return answers


def recheck_validation(spec: dict[str, Any], results: Path):
    """Re-grade the final-step validation dump against the protected answer key.

    Each dump row is matched to a key entry by its question; unknown, missing or repeated problems fail. The score
    uses the key's answers, never the dump's own `gts`.
    """
    relative = spec.get("validation_dump")
    if relative is None:
        return None, 0, []
    answers = load_answer_key(spec)
    root = results / spec["log_roots"][0]
    path = root / relative.format(step=spec["final_step"])
    label = path.relative_to(results).as_posix()
    if path.is_symlink() or not path.is_file():
        return None, 0, [f"missing validation dump {label}"]
    seen: dict[str, int] = {}
    unknown = 0
    correct = Decimal(0)
    # Split on \n only: dumps keep characters such as U+2028 unescaped, and str.splitlines() would break a row there.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        message = user_message(str(row.get("input", "")))
        key = question_key(message) if message is not None else None
        if key not in answers:
            unknown += 1
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            correct += Decimal(
                str(gsm8k_reference.compute_score(str(row.get("output", "")), answers[key]))
            )
    errors = []
    if unknown:
        errors.append(f"validation dump {label} has {unknown} rows matching no protected test problem")
    missing = len(answers) - len(seen)
    if missing:
        errors.append(f"validation dump {label} is missing {missing} of the {len(answers)} test problems")
    repeated = sum(n - 1 for n in seen.values())
    if repeated:
        errors.append(f"validation dump {label} repeats {repeated} test problems")
    if errors:
        return None, len(seen), errors
    return correct / Decimal(len(answers)), len(seen), []


def read_rows(path: Path) -> list[dict[str, Any]]:
    # Split on \n only: dumps keep characters such as U+2028 unescaped, and str.splitlines() would break a row there.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def check_group_diversity(spec: dict[str, Any], results: Path):
    """Per training step, how many GRPO groups (rows sharing a prompt) consist of identical responses.

    Every step from 1 to final_step must have a dump with `rollout_groups` prompts of `rollout_n` responses each, and
    fewer than `max_identical_group_share` of the groups may have all responses identical.
    """
    relative = spec.get("rollout_dump")
    if relative is None:
        return [], []
    root = results / spec["log_roots"][0]
    n = int(spec["rollout_n"])
    expected_groups = int(spec["rollout_groups"])
    limit = Decimal(str(spec["max_identical_group_share"]))
    steps, errors, failing = [], [], []
    for step in range(1, spec["final_step"] + 1):
        path = root / relative.format(step=step)
        label = path.relative_to(results).as_posix()
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing rollout dump {label}")
            continue
        groups: dict[str, list[str]] = {}
        for row in read_rows(path):
            groups.setdefault(str(row.get("input", "")), []).append(str(row.get("output", "")))
        if len(groups) != expected_groups or any(len(g) != n for g in groups.values()):
            errors.append(
                f"rollout dump {label} has {len(groups)} prompt groups with sizes "
                f"{sorted({len(g) for g in groups.values()})}; expected {expected_groups} groups of {n}"
            )
            continue
        identical = sum(len(set(g)) == 1 for g in groups.values())
        steps.append({"step": step, "groups": len(groups), "identical_groups": identical})
        if Decimal(identical) >= limit * len(groups):
            failing.append(steps[-1])
    if failing:
        first = failing[0]
        errors.append(
            f"{len(failing)} of {spec['final_step']} training steps have at least {limit} of their rollout groups "
            f"with all {n} responses identical (step {first['step']}: {first['identical_groups']} of "
            f"{first['groups']} groups)"
        )
    return steps, errors


def check_repeat(spec: dict[str, Any], results: Path):
    """The repeated run's rollout dumps must equal the main run's on `repeat_steps` (same multiset of rows)."""
    repeat = spec.get("repeat_root")
    if repeat is None:
        return [], []
    root = results / spec["log_roots"][0]
    compared, errors = [], []
    for step in spec["repeat_steps"]:
        main_path = root / spec["rollout_dump"].format(step=step)
        repeat_path = root / repeat / spec["rollout_dump"].format(step=step)
        missing = [p.relative_to(results).as_posix() for p in (main_path, repeat_path)
                   if p.is_symlink() or not p.is_file()]
        if missing:
            errors.append(f"missing rollout dump {', '.join(missing)} for the reproducibility check")
            continue
        main_rows = Counter(json.dumps(r, sort_keys=True) for r in read_rows(main_path))
        repeat_rows = Counter(json.dumps(r, sort_keys=True) for r in read_rows(repeat_path))
        differing = sum((main_rows - repeat_rows).values())
        total = sum(main_rows.values())
        compared.append({"step": step, "rows": total, "rows_not_reproduced": differing})
        if total == 0 or main_rows != repeat_rows:
            errors.append(
                f"rollout dump of step {step} is not reproduced by the repeated run: {differing} of {total} rows "
                f"differ (repeat has {sum(repeat_rows.values())} rows)"
            )
    return compared, errors


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
    recheck, recheck_rows, recheck_errors = recheck_validation(spec, results)
    errors.extend(recheck_errors)
    if recheck is not None:
        tolerance = Decimal(str(spec.get("validation_recheck_tolerance", "0.002")))
        if judged is not None and abs(recheck - judged) > tolerance:
            errors.append(
                f"logged {metric_key}={judged} disagrees with the verifier's re-grade "
                f"{recheck:.6f} of the step-{final_step} validation dump"
            )
        elif recheck < required_score:
            errors.append(
                f"re-graded step-{final_step} validation accuracy {recheck:.6f} is below "
                f"required score {required_score}"
            )
    diversity, diversity_errors = check_group_diversity(spec, results)
    errors.extend(diversity_errors)
    repeat, repeat_errors = check_repeat(spec, results)
    errors.extend(repeat_errors)
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
        "validation_recheck_score": float(recheck) if recheck is not None else None,
        "validation_recheck_rows": recheck_rows,
        "rollout_identical_groups": diversity,
        "repeat_comparison": repeat,
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
