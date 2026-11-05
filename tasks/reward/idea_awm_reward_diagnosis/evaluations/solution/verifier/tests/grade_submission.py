#!/usr/bin/env python3
"""Replay held-out episodes through the submission's repaired completion verifiers.

The agent's deliverable is workspace/data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl
with the defective completion-verification rules repaired in place. This grader:

1. loads that file and checks it is still a complete, parseable rule set
   (every original entry still present);
2. selects the rules for the replay episodes by (scenario, task_idx), exactly
   the way the training bridge does;
3. replays each held-out episode through the repaired rule: the audited
   false-positive episodes must no longer be judged complete, and synthesized
   correct completions of the same tasks must still be judged complete;
4. scores the fraction of correctly classified episodes.

There is no training rerun and no LLM judge: the verifier itself is the whole
evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent  # .../verifier
sys.path.insert(0, str(HERE / "resources" / "tools"))
import replay_fixtures as rf  # noqa: E402

VERIFIER_RELPATH = "data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl"

SPEC_KEYS = {
    "schema_version",
    "task_id",
    "grader",
    "cases_file",
    "required_entry_count",
    "required_score",
    "score_mode",
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
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("grader spec is not an object")
    unknown = set(spec) - SPEC_KEYS
    missing = SPEC_KEYS - set(spec)
    if unknown:
        raise ValueError(f"grader spec has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"grader spec is missing keys: {sorted(missing)}")
    if spec["score_mode"] != "final":
        raise ValueError("this grader only supports score_mode=final")
    return spec


def grade(spec: dict[str, Any], workspace: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cases = json.loads((HERE / "resources" / spec["cases_file"]).read_text(encoding="utf-8"))["cases"]

    verifier_file = workspace / VERIFIER_RELPATH
    if not verifier_file.is_file():
        raise ValueError(
            f"verifier file not found at {VERIFIER_RELPATH}: the completion verifiers must be repaired in place"
        )

    rules = rf.load_rules(verifier_file)
    if len(rules) != spec["required_entry_count"]:
        raise ValueError(
            f"verifier file now holds {len(rules)} entries, expected "
            f"{spec['required_entry_count']}: entries must be repaired in place, not added or removed"
        )

    rule_numbers = {}

    def get_code(scenario: str, task_idx: int):
        entry = rf.find_rule(rules, scenario, task_idx)
        if entry is None:
            raise KeyError(f"no rule for scenario={scenario} task_idx={task_idx}")
        code = entry.get("verification", {}).get("code")
        if not code:
            raise KeyError(f"rule scenario={scenario} task_idx={task_idx} has no verification code")
        return code, f"<submitted {scenario}:{task_idx}>"

    # Track which physical lines the replayed rules came from (diagnostics only).
    for number, entry in rules:
        key = (entry.get("scenario"), entry.get("task_idx"))
        rule_numbers.setdefault(key, number)

    results, errors = rf.replay_all(get_code, cases)

    total = len(cases)
    correct = sum(1 for r in results.values() if r["correct"])
    score = correct / total if total else 0.0
    required = float(spec["required_score"])
    passed = score >= required

    metrics: dict[str, Any] = {
        "task_id": spec["task_id"],
        "grader": spec["grader"],
        "verifier_file": VERIFIER_RELPATH,
        "verifier_entries": len(rules),
        "cases_total": total,
        "cases_correct": correct,
        "negatives_total": sum(1 for c in cases if c["class"] == "negative"),
        "negatives_correct": sum(
            1 for c in cases if c["class"] == "negative" and results[c["case_id"]]["correct"]
        ),
        "positives_total": sum(1 for c in cases if c["class"] == "positive"),
        "positives_correct": sum(
            1 for c in cases if c["class"] == "positive" and results[c["case_id"]]["correct"]
        ),
        "score": score,
        "required_score": required,
        "passed": passed,
        "case_results": [
            {
                "case_id": c["case_id"],
                "class": c["class"],
                "rule": f"{c['rule']['scenario']}:{c['rule']['task_idx']}",
                "verdict": results[c["case_id"]]["verdict"],
                "expected": results[c["case_id"]]["expected"],
                "correct": results[c["case_id"]]["correct"],
                "detail": results[c["case_id"]]["detail"],
            }
            for c in cases
        ],
        "errors": errors,
    }
    experiment = {
        "metrics": {
            "replay_score": score,
            "negatives_correct": metrics["negatives_correct"],
            "positives_correct": metrics["positives_correct"],
        }
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
    results.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"task_id": "idea_awm_reward_diagnosis", "errors": []}
    try:
        spec = load_spec(HERE / "resources" / "grader_spec.json")
        metrics["task_id"] = spec["task_id"]
        experiment, metrics, result = grade(spec, workspace)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        print(
            f"[{'PASS' if result['passed'] else 'FAIL'}] replay classification "
            f"{metrics['cases_correct']}/{metrics['cases_total']} "
            f"(score {result['score']:.4f}, required {metrics['required_score']})"
        )
        for case in metrics["case_results"]:
            if not case["correct"]:
                print(
                    f"  miss: {case['case_id']} rule={case['rule']} "
                    f"verdict={case['verdict']} expected={case['expected']}"
                )
        for error in metrics["errors"]:
            print(f"  error: {error}")
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        metrics["errors"].append(f"{type(exc).__name__}: {exc}")
        result = {"passed": False, "score": 0.0, "metrics": metrics}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", result)
        except OSError:
            pass
        print(f"[FAIL] {metrics['errors'][0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
