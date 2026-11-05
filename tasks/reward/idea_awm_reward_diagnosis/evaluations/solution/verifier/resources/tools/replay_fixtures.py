"""Shared fixture materialization and verifier replay for the solution grader.

Self-contained on purpose: the verifier stage ships only
evaluations/solution/, so this module re-implements the small pieces of the
issue-level curation tools that the replay needs (sqlite fixture
materialization, rule loading, and a per-case timeout guard).
"""

from __future__ import annotations

import json
import signal
import sqlite3
import tempfile
from pathlib import Path

CASE_TIMEOUT_S = 60


def materialize(path, tables):
    """Create a sqlite database at *path* from fixture {"table","ddl","rows"} dicts."""
    with sqlite3.connect(path) as conn:
        for table in tables:
            conn.execute(table["ddl"])
            if table["rows"]:
                placeholders = ",".join("?" for _ in table["rows"][0])
                conn.executemany(
                    'INSERT INTO "' + table["table"] + '" VALUES (' + placeholders + ")",
                    table["rows"],
                )


def load_rules(jsonl_path):
    """Yield (line_number, entry) for every parseable line of a verifier jsonl.

    Raises ValueError on the first malformed line: the training bridge loads
    this file wholesale, so any corruption is fatal regardless of which rule
    it touches.
    """
    rules = []
    with open(jsonl_path, "r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"verifier file line {number} is not valid JSON: {exc}") from exc
            rules.append((number, entry))
    if not rules:
        raise ValueError("verifier file contains no entries")
    return rules


def find_rule(rules, scenario, task_idx):
    """Mirror awm.tools.find_scenario_entry: first (scenario, task_idx) match."""
    for _, entry in rules:
        if entry.get("scenario") != scenario:
            continue
        if entry.get("task_idx") != task_idx:
            continue
        return entry
    return None


class _CaseTimeout(Exception):
    pass


def compile_verifier(code, source_name):
    """Exec a verifier entry's code and return its verify_task_completion."""
    namespace = {}
    exec(compile(code, source_name, "exec"), namespace)
    function = namespace.get("verify_task_completion")
    if not callable(function):
        raise ValueError("rule does not define verify_task_completion")
    return function


def _on_alarm(signum, frame):
    raise _CaseTimeout()


def run_case(verify, case, workdir):
    """Run one replay case; returns ("complete"|"incomplete"|"error", detail)."""
    casedir = Path(workdir) / case["case_id"]
    casedir.mkdir()
    initial = str(casedir / "initial.db")
    final = str(casedir / "final.db")
    materialize(initial, case["initial_tables"])
    materialize(final, case["final_tables"])
    answer = case.get("final_answer")
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(CASE_TIMEOUT_S)
    try:
        result = verify(initial, final, answer)
    except _CaseTimeout:
        return "error", "verifier timed out"
    except Exception as exc:  # a replay crash is a broken fix, not infra
        return "error", f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if not isinstance(result, dict):
        return "error", f"verifier returned {type(result).__name__}, expected dict"
    return ("complete" if result.get("result") == "complete" else "incomplete"), json.dumps(
        result, sort_keys=True
    )


def replay_all(get_code, cases):
    """Run every case through a rule-code lookup.

    get_code(scenario, task_idx) -> (code, source_name) or raises KeyError.
    Returns (results, errors): results maps case_id -> (verdict, expected,
    detail, error); errors collects load-level failures.
    """
    results = {}
    errors = []
    compiled = {}

    def verifier_for(scenario, task_idx):
        key = (scenario, task_idx)
        if key not in compiled:
            code, source_name = get_code(scenario, task_idx)
            compiled[key] = compile_verifier(code, source_name)
        return compiled[key]

    with tempfile.TemporaryDirectory(prefix="awm-replay-") as workdir:
        for case in cases:
            rule = case["rule"]
            expected = "complete" if case["expect"] == "accept" else "incomplete"
            try:
                verify = verifier_for(rule["scenario"], rule["task_idx"])
            except Exception as exc:
                verdict = "error"
                detail = f"rule load failed: {type(exc).__name__}: {exc}"
                errors.append(f"{case['case_id']}: {detail}")
            else:
                verdict, detail = run_case(verify, case, workdir)
                if verdict == "error":
                    errors.append(f"{case['case_id']}: {detail}")
            results[case["case_id"]] = {
                "verdict": verdict,
                "expected": expected,
                "correct": verdict == expected,
                "detail": detail,
            }
    return results, errors
