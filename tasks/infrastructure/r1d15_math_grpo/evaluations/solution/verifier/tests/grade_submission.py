#!/usr/bin/env python3
"""Grade the bounded scoring replay of the r1d15 math GRPO experiment.

The scoring run (resources/replay.sh, executed by the harness verifier stage or
the BATCH replay entry) restores the experiment's launch and instrumentation
pipeline, clears runs/run_001, and runs 30 training steps with validation and
checkpointing disabled. This grader reads only that fresh run's records:

  1. Structure  -- the run recorded exactly the required training steps, each
     with the run's per-step measurement records present and well-formed.
  2. Assets     -- the staged model and dataset files still match the task's
     content pins; the instrumentation files match their pristine copies.
  3. Configuration -- the run's effective (hydra-resolved) configuration matches
     the task's contract for every key the experiment's shape depends on.
  4. Measurement -- the run's token-weighted mean recorded measurement over the
     training steps is within the task's fixed bound.

Everything is deterministic; no judge model is involved. The score is binary.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "run_relpath",
    "metrics_filename",
    "required_steps",
    "metric_key",
    "max_token_weighted_mean",
    "hydra_glob",
    "config_contract",
    "pinned_files",
    "pinned_assets",
}

METRIC_FIELD = "parity/logprob_abs_mean_nat"
TOKEN_FIELD = "parity/tokens"
MAX_FIELD = "parity/logprob_abs_max_nat"


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
    if int(value["schema_version"]) != 1:
        raise ValueError("grader_spec.json schema_version must be 1")
    value["required_steps"] = int(value["required_steps"])
    value["max_token_weighted_mean"] = float(value["max_token_weighted_mean"])
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: Any) -> float | None:
    """Numeric value, or None for anything non-numeric (the recorder writes
    non-finite floats as strings; those are failures, not measurements)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_metric_rows(run_dir: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = run_dir / filename
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
                raise ValueError(f"{path}:{lineno}: malformed metrics row")
            rows.append(row)
    return rows


def training_steps(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Rows carrying the run's per-step training measurements, keyed by step."""
    steps: dict[int, dict[str, Any]] = {}
    for row in rows:
        metrics = row["metrics"]
        if METRIC_FIELD in metrics or TOKEN_FIELD in metrics or MAX_FIELD in metrics:
            step = row.get("step")
            if not isinstance(step, int):
                raise ValueError(f"training row with non-integer step: {step!r}")
            if step in steps:
                raise ValueError(f"duplicate training row for step {step}")
            steps[step] = metrics
    return steps


def check_structure(
    steps: dict[int, dict[str, Any]], required: int, errors: list[str]
) -> list[float]:
    """Steps 1..required exactly, each with well-formed measurements."""
    weighted: list[float] = []
    expected = set(range(1, required + 1))
    found = set(steps)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing:
            errors.append(
                f"the scoring run recorded {len(found)} of the required {required} "
                f"training steps; missing steps: {missing[:12]}"
            )
        if extra:
            errors.append(
                f"the scoring run recorded unexpected training steps: {extra[:12]}"
            )
        return weighted
    for step in sorted(steps):
        metrics = steps[step]
        tokens = as_float(metrics.get(TOKEN_FIELD))
        mean = as_float(metrics.get(METRIC_FIELD))
        peak = as_float(metrics.get(MAX_FIELD))
        if tokens is None or tokens <= 0:
            errors.append(f"step {step}: no generated tokens recorded")
            continue
        if mean is None or peak is None:
            errors.append(
                f"step {step}: recorded measurement is missing or non-finite"
            )
            continue
        weighted.append(mean * tokens)
    return weighted


def check_pinned(
    workspace: Path, pins: dict[str, str], label: str, errors: list[str]
) -> dict[str, str]:
    violations: dict[str, str] = {}
    for rel, want in sorted(pins.items()):
        target = workspace / rel
        if not target.is_file():
            violations[rel] = "missing"
            errors.append(f"{label} {rel} is missing from the staged workspace")
            continue
        got = sha256_file(target)
        if got != want:
            violations[rel] = f"sha256 {got}"
            errors.append(f"{label} {rel} does not match the task's content pin")
    return violations


def dig(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


_INTERP_RE = re.compile(r"^\$\{([a-zA-Z0-9_.]+)\}$")


def resolve_interpolation(
    cfg: dict[str, Any], value: Any, depth: int = 0
) -> Any:
    """Resolve a hydra/omegaconf config-dump interpolation reference.

    Hydra's .hydra/config.yaml dump keeps "${a.b.c}" interpolations unexpanded;
    the effective value is the referenced key's value in the same dump. Only
    plain dotted references are followed (no oc.select/expr forms, which stay
    as-is); unresolvable references keep the raw string so the violation
    remains visible.
    """
    if depth > 4 or not isinstance(value, str):
        return value
    m = _INTERP_RE.match(value)
    if not m:
        return value
    try:
        return resolve_interpolation(cfg, dig(cfg, m.group(1)), depth + 1)
    except KeyError:
        return value


def check_config(
    run_dir: Path, spec: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    import yaml

    configs = sorted(glob.glob(str(run_dir / spec["hydra_glob"])))
    if not configs:
        errors.append(
            "the scoring run recorded no effective configuration "
            f"({spec['hydra_glob']} under {spec['run_relpath']})"
        )
        return None
    if len(configs) > 1:
        errors.append(
            f"the scoring run recorded {len(configs)} effective configurations; "
            "exactly one scoring attempt is expected"
        )
        return None
    with open(configs[0], encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    violations: dict[str, Any] = {}
    for key, want in sorted(spec["config_contract"].items()):
        try:
            got = resolve_interpolation(cfg, dig(cfg, key))
        except KeyError:
            violations[key] = "missing"
            errors.append(f"effective configuration is missing {key}")
            continue
        if got != want:
            violations[key] = {"expected": want, "got": got}
            errors.append(
                f"effective configuration changed {key}: expected {want!r}, got {got!r}"
            )
    for key in spec.get("config_free_keys", []):
        try:
            dig(cfg, key)
        except KeyError:
            violations[key] = "missing"
            errors.append(f"effective configuration is missing {key}")
    return violations


def grade(spec: dict[str, Any], workspace: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    run_dir = workspace / spec["run_relpath"]
    required = spec["required_steps"]

    steps: dict[int, dict[str, Any]] = {}
    if (run_dir / spec["metrics_filename"]).is_file():
        rows = load_metric_rows(run_dir, spec["metrics_filename"])
        steps = training_steps(rows)
    else:
        errors.append(
            f"the scoring run produced no {spec['metrics_filename']} under "
            f"{spec['run_relpath']}; there is nothing to grade"
        )

    weighted = check_structure(steps, required, errors)

    total_tokens = sum(as_float(m.get(TOKEN_FIELD)) or 0.0 for m in steps.values())
    token_weighted_mean: float | None = None
    if weighted:
        token_weights = sum(as_float(m.get(TOKEN_FIELD)) or 0.0 for m in steps.values())
        token_weighted_mean = sum(weighted) / token_weights
    if token_weighted_mean is not None:
        bound = spec["max_token_weighted_mean"]
        if token_weighted_mean > bound:
            errors.append(
                f"token-weighted mean of {METRIC_FIELD} over the scoring run's "
                f"{len(steps)} training steps is {token_weighted_mean:.6f}, "
                f"above the required bound {bound}"
            )

    config_violations = check_config(run_dir, spec, errors)
    file_violations = check_pinned(workspace, spec["pinned_files"], "instrumentation file", errors)
    asset_violations = check_pinned(workspace, spec["pinned_assets"], "staged asset", errors)

    means = [as_float(m.get(METRIC_FIELD)) for m in steps.values()]
    means = [m for m in means if m is not None]
    peaks = [as_float(m.get(MAX_FIELD)) for m in steps.values()]
    peaks = [p for p in peaks if p is not None]

    passed = not errors
    score = 1.0 if passed else 0.0
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": spec["metric_key"],
        "score_mode": "final",
        "required_steps": required,
        "recorded_steps": len(steps),
        "token_weighted_mean": token_weighted_mean,
        "max_token_weighted_mean": spec["max_token_weighted_mean"],
        "best_step": (required if passed else None),
        "best_score": (token_weighted_mean if token_weighted_mean is not None else None),
        "final_score": (token_weighted_mean if token_weighted_mean is not None else None),
        "per_step_mean_min": min(means) if means else None,
        "per_step_mean_max": max(means) if means else None,
        "per_step_peak_max": max(peaks) if peaks else None,
        "total_tokens": total_tokens,
        "config_violations": config_violations,
        "file_violations": file_violations,
        "asset_violations": asset_violations,
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"run_dir": spec["run_relpath"]},
        "metrics": {
            "metric_key": spec["metric_key"],
            "metric_status": "available" if token_weighted_mean is not None else "missing",
            "best_step": metrics["best_step"],
            "best_score": metrics["best_score"],
            "final_score": metrics["final_score"],
            "required_steps": required,
            "recorded_steps": len(steps),
        },
    }
    return experiment, metrics, {"passed": passed, "score": score, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")),
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    results = args.results.expanduser().resolve()
    try:
        spec = load_spec(Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json")
        experiment, metrics, result = grade(spec, workspace)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
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
    sys.exit(main())
