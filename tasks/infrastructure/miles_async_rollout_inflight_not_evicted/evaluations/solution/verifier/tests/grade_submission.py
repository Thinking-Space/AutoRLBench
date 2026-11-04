#!/usr/bin/env python3
"""Grade a fresh fully-async GRPO rerun for the abort-eviction task.

Two signals, both read only from $AUTORLBENCH_RESULTS/experiment_logs:

  * train-step throughput (primary): how many training steps the fixed
    wall-clock budget completed, counted from the driver log's ``rollout N:``
    lines. The defect keeps interrupted in-flight requests alive across each
    weight sync, so the engine wastes full-length decode on trajectories that
    are later dropped as stale, and far fewer steps finish in the same wall
    clock (historical 92 defective vs 148 control).
  * last-window solve rate (secondary): the mean fraction of samples with a
    correct answer (``reward.acc``) over the last ``solve_window_frac`` of the
    dumped per-step rollout data. It is throughput-driven -- more completed
    steps means more learning by the end of the fixed window -- so it corroborates
    the throughput signal rather than standing alone.

Plus two guards, so the numbers cannot be satisfied without doing the work:

  * the launch recipe is fixed by the instruction, and the grader checks it
    from the fresh run's own logs -- the ray entrypoint line must still carry
    the configured argv tokens and the resolved config echo must agree
    (raising ``--max-weight-staleness`` or switching the pause mode restores
    throughput without touching the defect, so it must fail).
  * the training window (first log timestamp to the last step line's own
    timestamp) must stay within ``max_log_span_seconds`` -- a run stretched
    past the fixed budget trades wall clock for steps and must fail.

Pass requires BOTH thresholds and both guards. Score is 1.0 or 0.0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
# matches e.g. "log_utils.py:66 - rollout 17: {'rollout/response_lengths': ...}"
STEP_RE = re.compile(r"\brollout\s+(\d+):\s*\{", re.IGNORECASE)
# matches both driver-log styles: "2026-10-10 14:27:01,098 INFO ..." and
# "[2026-10-10 18:59:32.567 actor_cell0_rank0] log_utils.py:66 - ..."
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
MAX_FILE_BYTES = 256 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "log_roots",
    "step_metric_key",
    "required_train_steps",
    "solve_rate_metric_key",
    "solve_window_frac",
    "required_solve_rate",
    "required_recipe_tokens",
    "required_echo",
    "max_log_span_seconds",
}


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
    if not REQUIRED_SPEC_KEYS <= set(value):
        raise ValueError("grader_spec.json is missing required fields")
    if value["schema_version"] != 1:
        raise ValueError("unsupported grader spec")
    if value["log_roots"] != ["experiment_logs"]:
        raise ValueError("log roots must be experiment_logs")
    if not isinstance(value["required_recipe_tokens"], list) or not all(
        isinstance(t, str) for t in value["required_recipe_tokens"]
    ):
        raise ValueError("required_recipe_tokens must be a list of strings")
    if not isinstance(value["required_echo"], dict):
        raise ValueError("required_echo must be an object")
    if _dec(value["max_log_span_seconds"]) <= 0:
        raise ValueError("max_log_span_seconds must be positive")
    return value


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc


def count_train_steps(results: Path, roots: list[str]) -> tuple[int, list[str]]:
    """Largest ``rollout N`` index + 1 across all text logs under the roots."""
    seen: set[int] = set()
    sources: list[str] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in {".log", ".txt", ".out", ""}:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            text = ANSI_RE.sub("", data.decode("utf-8", errors="replace"))
            hits = [int(m.group(1)) for m in STEP_RE.finditer(text)]
            if hits:
                seen.update(hits)
                sources.append(path.relative_to(results).as_posix())
    steps = (max(seen) + 1) if seen else 0
    return steps, sorted(set(sources))


def collect_log_texts(results: Path, roots: list[str]) -> list[tuple[Path, str]]:
    """Every text log under the roots, ANSI-stripped: the same file set (and the
    same filters) the step counter scans, reused by the launch-recipe checks."""
    texts: list[tuple[Path, str]] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in {".log", ".txt", ".out", ""}:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            texts.append((path, ANSI_RE.sub("", data.decode("utf-8", errors="replace"))))
    return texts


def _parse_ts(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def train_window_seconds(texts: list[tuple[Path, str]]) -> float | None:
    """Wall-clock span of the training window: the first ``rollout N:`` step
    line's own timestamp to the last one's.

    Deliberately only the step lines, not the whole log: the driver log mixes
    clocks (the sglang-router's Rust lines carry UTC timestamps while the
    Python side logs local time), and teardown after the budget SIGINT can
    crawl -- either would inflate the span of a legitimate run. The step lines
    all come from one logger (log_utils.py:66), so their timestamps share one
    clock, and a run stretched past the budget shows up as step lines
    continuing to appear late.
    """
    first: datetime | None = None
    last: datetime | None = None
    for _path, text in texts:
        for line in text.splitlines():
            if not STEP_RE.search(line):
                continue
            for stamp in TS_RE.findall(line):
                ts = _parse_ts(stamp)
                if ts is None:
                    continue
                if first is None or ts < first:
                    first = ts
                if last is None or ts > last:
                    last = ts
    if first is None or last is None:
        return None
    return (last - first).total_seconds()


def check_launch_recipe(
    results: Path, roots: list[str], spec: dict[str, Any]
) -> tuple[list[str], float | None]:
    """The fixed dataset / model / rollout configuration / wall-clock budget,
    checked against the fresh run's own logs.

    Two independent sources, both written by the run itself: the ray job
    entrypoint line echoes the full submitted argv, and the driver's config
    dump echoes every resolved setting. A repair that instead relaxes the
    recipe -- raising ``--max-weight-staleness`` so interrupted groups train
    rather than get dropped, switching the pause mode, shrinking the response
    length -- restores the graded numbers without touching the defect, so any
    token that disappears, any echoed value that changes, or a training window
    past the span cap fails the run.

    Every echo of a key must agree with the spec (not just the first match):
    a duplicated argv where the later value wins resolves in the echo, and a
    stray earlier line must not mask a changed one.
    """
    texts = collect_log_texts(results, roots)
    if not texts:
        return ["no text logs under experiment_logs to check the launch recipe"], None
    joined = "\n".join(text for _path, text in texts)

    errors: list[str] = []
    for token in spec["required_recipe_tokens"]:
        if token not in joined:
            errors.append(f"fixed launch recipe changed: {token!r} missing from the run logs")
    for key, expected in spec["required_echo"].items():
        pattern = rf"^[ \t]*{re.escape(str(key))}[ .\t]+(\S+)[ \t\r]*$"
        matches = re.findall(pattern, joined, re.MULTILINE)
        if not matches:
            errors.append(f"fixed launch recipe changed: resolved {key} not echoed in the run logs")
        elif any(v != str(expected) for v in matches):
            seen = sorted(set(matches))
            errors.append(f"fixed launch recipe changed: {key} echoed as {seen}, expected {expected}")

    span = train_window_seconds(texts)
    if span is not None:
        cap = _dec(spec["max_log_span_seconds"])
        if _dec(span) > cap:
            errors.append(
                f"run exceeded the fixed wall-clock budget: training window {span:.0f}s > {cap}s"
            )
    return errors, span


def _iter_samples(node: Any):
    if isinstance(node, list):
        for child in node:
            yield from _iter_samples(child)
    else:
        yield node


def solve_rate_window(
    results: Path, roots: list[str], window_frac: Decimal
) -> tuple[Decimal | None, dict[str, Any]]:
    """Mean solve rate over the last ``window_frac`` of dumped rollout steps."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(f"torch unavailable for solve-rate grading: {exc}") from exc

    pt_files: list[Path] = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("dump/rollout_data/*.pt"):
            if path.is_file() and not path.is_symlink():
                pt_files.append(path)

    def step_of(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return -1

    pt_files = sorted((p for p in pt_files if step_of(p) >= 0), key=step_of)
    per_step: list[dict[str, Any]] = []
    for path in pt_files:
        try:
            dumped = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a corrupt dump should not crash grading
            continue
        samples = dumped.get("samples") if isinstance(dumped, dict) else None
        if samples is None:
            continue
        accs = []
        for sample in _iter_samples(samples):
            if not isinstance(sample, dict):
                continue
            reward = sample.get("reward")
            if isinstance(reward, dict):
                accs.append(1.0 if reward.get("acc") else 0.0)
        if accs:
            per_step.append({"step": step_of(path), "n": len(accs),
                             "solve_rate": sum(accs) / len(accs)})

    info: dict[str, Any] = {
        "dump_files": len(pt_files),
        "scored_steps": len(per_step),
        "window_frac": float(window_frac),
    }
    if not per_step:
        info["window_steps"] = []
        return None, info
    k = max(1, int(len(per_step) * float(window_frac)))
    window = per_step[-k:]
    info["window_steps"] = [s["step"] for s in window]
    mean = Decimal(sum(s["solve_rate"] for s in window)) / Decimal(len(window))
    return mean, info


def grade(spec: dict[str, Any], results: Path):
    roots = spec["log_roots"]
    required_steps = int(_dec(spec["required_train_steps"]))
    required_solve = _dec(spec["required_solve_rate"])
    window_frac = _dec(spec["solve_window_frac"])

    train_steps, step_sources = count_train_steps(results, roots)
    solve_rate, solve_info = solve_rate_window(results, roots, window_frac)
    recipe_errors, log_span = check_launch_recipe(results, roots, spec)

    errors: list[str] = []
    if train_steps <= 0:
        errors.append(
            f"no {spec['step_metric_key']} found in experiment_logs "
            "(expected driver log 'rollout N:' lines)"
        )
    elif train_steps < required_steps:
        errors.append(
            f"train_steps {train_steps} is below required {required_steps}"
        )
    if solve_rate is None:
        errors.append(
            f"no {spec['solve_rate_metric_key']} found "
            "(expected dump/rollout_data/*.pt with reward.acc)"
        )
    elif solve_rate < required_solve:
        errors.append(
            f"solve_rate {solve_rate} is below required {required_solve}"
        )
    errors.extend(recipe_errors)

    passed = not errors
    score = 1.0 if passed else 0.0
    metrics = {
        "task_id": spec["task_id"],
        "step_metric_key": spec["step_metric_key"],
        "train_steps": train_steps,
        "required_train_steps": required_steps,
        "train_steps_passed": train_steps >= required_steps and train_steps > 0,
        "solve_rate_metric_key": spec["solve_rate_metric_key"],
        "solve_rate": float(solve_rate) if solve_rate is not None else None,
        "required_solve_rate": float(required_solve),
        "solve_rate_passed": solve_rate is not None and solve_rate >= required_solve,
        "solve_rate_info": solve_info,
        "step_source_logs": step_sources,
        "recipe_errors": recipe_errors,
        "train_window_seconds": round(log_span, 1) if log_span is not None else None,
        "max_log_span_seconds": float(_dec(spec["max_log_span_seconds"])),
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": step_sources},
        "metrics": {
            "train_steps": train_steps,
            "solve_rate_last_window": float(solve_rate) if solve_rate is not None else None,
            "solve_rate_info": solve_info,
        },
    }
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
