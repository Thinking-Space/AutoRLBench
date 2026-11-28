#!/usr/bin/env python3
"""Grade the fresh tiny-MoE rerun and enforce router integrity invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(results: Path) -> list[dict[str, Any]]:
    root = results / "experiment_logs"
    rows: list[dict[str, Any]] = []
    if not root.is_dir() or root.is_symlink():
        return rows
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_symlink() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def grade(spec: dict[str, Any], results: Path, workspace: Path) -> dict[str, Any]:
    errors: list[str] = []
    report = workspace / "report.md"
    if report.is_symlink() or not report.is_file() or not report.read_text(encoding="utf-8").strip():
        errors.append("missing or empty report.md")

    for relative, expected in spec["protected_files"].items():
        path = workspace / relative
        if path.is_symlink() or not path.is_file():
            errors.append(f"protected file missing or replaced: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"protected file modified: {relative}")

    rows = load_rows(results)
    final_rows = [row for row in rows if row.get("step") == spec["final_step"]]
    if len(final_rows) != 1:
        errors.append(f"expected exactly one step-{spec['final_step']} row, found {len(final_rows)}")
        final = {}
    else:
        final = final_rows[0]
    if any(isinstance(row.get("step"), int) and row["step"] > spec["final_step"] for row in rows):
        errors.append("found metrics beyond declared step budget")

    score: Decimal | None = None
    try:
        score = Decimal(str(final[spec["metric_key"]]))
        if not score.is_finite():
            raise InvalidOperation
    except (KeyError, InvalidOperation):
        errors.append(f"missing finite {spec['metric_key']} at final step")
    if score is not None:
        if score < Decimal(spec["required_final_score"]):
            errors.append(f"final probability {score} is below {spec['required_final_score']}")
        if score > Decimal(spec["required_final_score_max"]):
            errors.append(f"final probability {score} exceeds {spec['required_final_score_max']}")

    checks = {
        "router/bias_dtype": spec["required_router_dtype"],
        "router/bias_unique_values": spec["required_router_unique_values"],
        "router/selected_expert": spec["required_selected_expert"],
    }
    for key, expected in checks.items():
        if final.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}, got {final.get(key)!r}")
    router_hash = final.get("router/bias_sha256")
    source_hash = final.get("source/bias_sha256")
    expected_hash = spec["required_source_hash"]
    if source_hash != expected_hash or router_hash != expected_hash:
        errors.append(
            "router/source bias SHA-256 values must both match the expected source hash"
        )

    passed = not errors
    metrics = {
        "task_id": spec["task_id"],
        "final_step": spec["final_step"],
        "final_score": float(score) if score is not None else None,
        "router_dtype": final.get("router/bias_dtype"),
        "router_unique_values": final.get("router/bias_unique_values"),
        "selected_expert": final.get("router/selected_expert"),
        "source_hash_match": router_hash == source_hash if router_hash else False,
        "errors": errors,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "deterministic": True,
        "judge_used": False,
    }
    return {"passed": passed, "score": metrics["score"], "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace")))
    args = parser.parse_args()
    results = args.results.resolve()
    spec_path = Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result = grade(spec, results, args.workspace.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"passed": False, "score": 0.0, "metrics": {"errors": [f"verifier error: {exc}"]}}
    write_json(results / "metrics.json", result["metrics"])
    write_json(results / "result.json", result)
    if result["metrics"].get("errors"):
        print("; ".join(result["metrics"]["errors"]), file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
