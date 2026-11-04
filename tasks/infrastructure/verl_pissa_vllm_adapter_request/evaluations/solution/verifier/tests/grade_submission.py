#!/usr/bin/env python3
"""Extract the target veRL metric from fresh logs and assign a binary score.

The graded metric is `val-health/actor_rollout_policy_agreement` at the final
step, computed by instrumentation that lives in the shipped workspace, so a
bare log line proves nothing. The score is therefore anchored four ways; only
the shapes and cross-relations of a real run are accepted, never a value on
its own:

* verl's file log (`VERL_FILE_LOGGER_PATH`, one `{"step", "data"}` JSON row
  per training step 0..final_step) must show a complete run, and its final
  row must carry the mae/agreement pair with agreement == exp(-mae); the
  value the text log was judged on must equal the file-log value at
  final_step.
* the rollout sync probe dump must still show the PiSSA actor (`is_lora`,
  `is_svd_init`, `rollout_sync_mode` `lora_delta`, the PiSSA adapter path),
  and the per-request generate dumps the server writes must mostly show
  `lora_request_used` -- together these reject the masked repair of
  disabling the adapter everywhere, which would trivially agree at ~1.0
  while training the residual base only.
* live grading (the stage sets AUTORLBENCH_WORKSPACE / AUTORLBENCH_VERIFIER)
  sha256-pins the runner and the health-instrumentation file against
  grader_spec.json, so editing the budget or the metric computation fails;
  the repair itself lives in the rollout request path and is not pinned.

A replay of packaged evidence by hand runs without the stage environment and
skips the workspace checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
MAX_PROBE_FILES = 256
TRANSFORM_TOLERANCE = 1e-9
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "pinned_sha256",
}
OPTIONAL_SPEC_KEYS = {"required_final_score_max", "score_mode"}
METRIC_KEY = "val-health/actor_rollout_policy_agreement"
MAE_KEY = "val-health/actor_rollout_policy_mae"
VAL_REWARD_KEY = "val-aux/openai/gsm8k/reward/mean@1"
TRAIN_KEYS = ("actor/grad_norm", "timing_s/step")
RUNNER_ARTIFACTS = ("source_commit.txt", "environment-freeze.txt", "nvidia-smi.txt")


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
    if not isinstance(value["pinned_sha256"], dict) or not value["pinned_sha256"]:
        raise ValueError("grader_spec.json must pin at least one workspace file")
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


def probe_files(results: Path, roots: list[str], name_pattern: str) -> list[Path]:
    """JSON probe dumps under the log roots, regular files only, newest first."""
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob(name_pattern):
            if path.is_symlink() or not path.is_file():
                continue
            if "wandb" in path.relative_to(root).parts:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


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


def find_step_log(results: Path, roots: list[str]) -> Path | None:
    """verl's file logger: `<log root>/metrics/all_steps.jsonl`, newest, wandb-free."""
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("all_steps.jsonl"):
            if path.is_symlink() or not path.is_file():
                continue
            if "wandb" in path.relative_to(root).parts:
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)[0] if found else None


def read_json_rows(path: Path) -> list[dict[str, Any]] | str:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return f"line {lineno}: not JSON"
            if isinstance(row, dict):
                rows.append(row)
    return rows


def training_reality(
    results: Path, spec: dict[str, Any], judged: Decimal | None
) -> tuple[bool, dict[str, Any], str | None]:
    """A run that actually executed the shipped runner leaves a fixed trail:
    verl's file logger with one row per step, the runner's own artifacts, the
    rollout dumps, and probe JSONs whose cross-relations hold. Only shapes and
    cross-relations are checked here; the pass/fail threshold is applied to the
    judged metric value, never to these artifacts."""
    final_step = spec["final_step"]
    details: dict[str, Any] = {
        "step_log": None,
        "metrics_rows": 0,
        "steps_continuous": False,
        "final_health_pair": False,
        "transform_consistent": False,
        "text_matches_file_log": None,
        "runner_artifacts": {},
        "generation_dumps": 0,
        "lora_request_used": 0,
        "sync_dump": None,
        "sync_dump_fields": {},
    }
    path = find_step_log(results, spec["log_roots"])
    if path is None:
        return False, details, (
            "missing metrics/all_steps.jsonl under "
            f"{spec['log_roots']} (the shipped runner sets VERL_FILE_LOGGER_PATH and "
            "verl writes one JSON row per training step; a run that did not execute "
            "it cannot have produced the judged log line)"
        )
    details["step_log"] = path.relative_to(results).as_posix()
    rows = read_json_rows(path)
    if isinstance(rows, str):
        return False, details, f"{details['step_log']}: malformed training log ({rows})"
    details["metrics_rows"] = len(rows)
    steps = [row["step"] for row in rows if isinstance(row.get("step"), int)]
    details["steps_continuous"] = steps == list(range(final_step + 1))
    if not details["steps_continuous"]:
        return False, details, (
            f"{details['step_log']} does not show a complete run: expected one row per "
            f"training step 0..{final_step} (verl's file logger), got {len(steps)} rows"
        )
    for row in rows:
        step = row["step"]
        data = row.get("data")
        if not isinstance(data, dict):
            return False, details, f"{details['step_log']} step {step}: no metric payload"
        if VAL_REWARD_KEY not in data:
            return False, details, (
                f"{details['step_log']} step {step}: no {VAL_REWARD_KEY} row (the "
                "runner evaluates every step with trainer.test_freq=1)"
            )
        if step >= 1:
            for key in TRAIN_KEYS:
                if key not in data:
                    return False, details, (
                        f"{details['step_log']} step {step}: no {key} (the runner's "
                        "four GRPO updates must be logged)"
                    )
    final_data = next(row["data"] for row in rows if row["step"] == final_step)
    mae = final_data.get(MAE_KEY)
    agreement = final_data.get(METRIC_KEY)
    details["final_health_pair"] = (
        isinstance(mae, (int, float)) and isinstance(agreement, (int, float))
    )
    if not details["final_health_pair"]:
        return False, details, (
            f"{details['step_log']} step {final_step}: the final health pair "
            f"({MAE_KEY}, {METRIC_KEY}) is missing; the runner's "
            "log_final_policy_health wrote neither, so the judged line is not "
            "backed by the file logger"
        )
    details["transform_consistent"] = (
        abs(float(agreement) - math.exp(-float(mae))) <= TRANSFORM_TOLERANCE
    )
    if not details["transform_consistent"]:
        return False, details, (
            f"{details['step_log']} step {final_step}: agreement {agreement} does not "
            f"equal exp(-mae) with mae {mae}; the two instrumented values must "
            "satisfy the logged transform"
        )
    if judged is not None:
        details["text_matches_file_log"] = abs(float(judged) - float(agreement)) <= TRANSFORM_TOLERANCE
        if details["text_matches_file_log"] is False:
            return False, details, (
                f"the judged text-log value {judged} at step {final_step} disagrees "
                f"with {details['step_log']} ({agreement}); a real run logs the same "
                "value through both sinks"
            )
    for name in RUNNER_ARTIFACTS:
        artifact = results / spec["log_roots"][0] / name
        details["runner_artifacts"][name] = artifact.is_file() and not artifact.is_symlink()
    missing_artifacts = [n for n, ok in details["runner_artifacts"].items() if not ok]
    if missing_artifacts:
        return False, details, (
            f"missing runner artifacts under {spec['log_roots'][0]}: "
            f"{', '.join(missing_artifacts)} (the shipped runner writes them "
            "before training starts)"
        )
    for subdir, what in (("rollouts", "rollout dumps"), ("validation_generations", "validation dumps")):
        folder = results / spec["log_roots"][0] / subdir
        has = folder.is_dir() and not folder.is_symlink() and any(
            p.is_file() and not p.is_symlink() and p.suffix == ".jsonl" for p in folder.iterdir()
        )
        if not has:
            return False, details, (
                f"missing {what} under {spec['log_roots'][0]}/{subdir} (the runner "
                f"sets trainer.{'rollout_data_dir' if subdir == 'rollouts' else 'validation_data_dir'} there)"
            )
    sync_dumps = probe_files(results, spec["log_roots"], "*.rollout_sync_latest*.json")
    if not sync_dumps:
        return False, details, (
            "missing rollout sync probe dump (*rollout_sync_latest*.json) under "
            f"{spec['log_roots']} (the shipped instrumentation writes it per sync; a "
            "run without it did not use the shipped rollout worker)"
        )
    try:
        sync_payload = json.loads(sync_dumps[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, details, f"unreadable rollout sync probe dump: {exc}"
    details["sync_dump"] = sync_dumps[0].relative_to(results).as_posix()
    details["sync_dump_fields"] = {
        "is_lora": sync_payload.get("is_lora"),
        "is_svd_init": sync_payload.get("is_svd_init"),
        "rollout_sync_mode": sync_payload.get("rollout_sync_mode"),
        "lora_adapter_path": sync_payload.get("lora_adapter_path"),
    }
    if not (
        sync_payload.get("is_lora") is True
        and sync_payload.get("is_svd_init") is True
        and sync_payload.get("rollout_sync_mode") == "lora_delta"
        and "pissa_adapter" in str(sync_payload.get("lora_adapter_path", ""))
    ):
        return False, details, (
            f"{details['sync_dump']} does not show the PiSSA actor (is_lora/"
            "is_svd_init/rollout_sync_mode=lora_delta/PiSSA adapter path); the "
            "repair must attach the adapter to rollout requests, not remove it "
            "from the actor"
        )
    generate_dumps = probe_files(results, spec["log_roots"], "*.generate*.json")
    used = total = 0
    for dump in generate_dumps[:MAX_PROBE_FILES]:
        try:
            payload = json.loads(dump.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("lora_request_used"), bool):
            total += 1
            used += int(payload["lora_request_used"])
    details["generation_dumps"] = total
    details["lora_request_used"] = used
    if total == 0:
        return False, details, (
            "missing generate probe dumps (*.generate*.json) under "
            f"{spec['log_roots']} (the shipped server writes one per request; a run "
            "without them did not use the shipped rollout request path)"
        )
    if used * 2 <= total:
        return False, details, (
            f"only {used}/{total} generate probe dumps show lora_request_used; the "
            "repair must attach the loaded adapter to the vLLM generation requests, "
            "not agree with the actor by disabling it on both sides"
        )
    return True, details, None


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    """The rerun stage executes the agent's workspace, so live grading
    sha256-pins the files a legitimate repair does not touch: the runner
    (budget contract) and the health instrumentation (metric computation).
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
    reality_ok, reality_details, reality_error = training_reality(results, spec, judged)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the workspace as shipped apart from the repair: "
            "the runner and the health instrumentation are sha256-pinned at "
            "grading time, so editing the budget, the launch script or the metric "
            "computation fails"
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
        "training_reality": reality_details,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": integrity_details,
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
