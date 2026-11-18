"""Grade a repaired pool gate.

Two passes decide the score:

* the system's rerun of `scripts/run_cases.py` over the staged case list, read from the fresh
  run's `experiment_logs/cases.jsonl`, and
* a held-out pass this verifier drives itself: the same entry point, the repaired launcher, and
  a case list the workspace never contained, over pools the workspace already has.

Each case is turned into an outcome -- `accept`, `reject` or `unsafe-accept` -- from the run's
own record and the pool on disk, never from what the gate said about itself, and compared with
the reference outcome. A gate passes when it reproduces every reference outcome on both lists,
never accepts unsafely, and never rejects a pool the builder legitimately produces.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1]
RESOURCES = VERIFIER / "resources"
OUTCOMES = ("accept", "reject", "unsafe-accept")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- outcome derivation
def pool_facts(workspace: Path, pool_dir: str) -> dict:
    pool = workspace / pool_dir
    rows = read_jsonl(pool / "queries.jsonl")
    identities = [(row["metadata"]["pool_source"], row["metadata"]["instance_id"]) for row in rows]
    counts: dict[str, int] = {}
    for source, _ in identities:
        counts[source] = counts.get(source, 0) + 1
    return {"rows": len(rows), "unique": len(set(identities)), "source_counts": dict(sorted(counts.items()))}


def tag_total(run_tag: str) -> str | None:
    match = re.match(r"^mixed([^_]*)_", run_tag or "")
    return match.group(1) if match else None


def outcome(record: dict, facts: dict, case: dict | None = None) -> tuple[str, list[str]]:
    if not record.get("accepted"):
        return "reject", []
    run = record.get("run")
    if run is None:
        return "accept", []
    problems = []
    # A run tag the launch case pins is the operator's to choose; the gate only has to leave it
    # alone. Everything else about the run is still checked.
    pinned_tag = (case or {}).get("env", {}).get("RUN_TAG")
    if pinned_tag:
        if run.get("run_tag") != pinned_tag:
            problems.append("the gate replaced the run tag the case pinned")
        claimed = None
    else:
        claimed = tag_total(run.get("run_tag", ""))
    if pinned_tag:
        pass
    elif not claimed:
        problems.append("run identity has no pool-size component")
    elif not claimed.isdigit():
        problems.append(f"run identity carries a non-integer pool size: {claimed!r}")
    elif int(claimed) != facts["rows"]:
        problems.append(f"run identity claims {claimed} queries, the pool holds {facts['rows']}")
    total_env = str(run.get("pool_total_env", ""))
    if not total_env.strip():
        problems.append("the run was given no pool total")
    elif not total_env.strip().isdigit():
        problems.append(f"the run was given a non-integer pool total: {total_env!r}")
    elif int(total_env) != facts["rows"]:
        problems.append(f"the run was given total={total_env}, the pool holds {facts['rows']}")
    if run.get("rows_read") != facts["rows"] or run.get("unique_rows") != facts["unique"]:
        problems.append("the run did not read the pool the gate validated")
    if run.get("pool_source_counts") != facts["source_counts"]:
        problems.append("the run saw a different source mix than the pool holds")
    config = run.get("rollout_config", {})
    over, batch = config.get("over_sampling_batch_size"), config.get("rollout_batch_size")
    if isinstance(over, int) and over > facts["rows"]:
        problems.append(f"over-sampling batch {over} exceeds the pool's {facts['rows']} queries")
    if isinstance(over, int) and isinstance(batch, int) and over < batch:
        problems.append(f"over-sampling batch {over} is smaller than the rollout batch {batch}")
    if run.get("duplicate_prompts_total"):
        problems.append(f"{run['duplicate_prompts_total']} prompts repeated inside a rollout")
    return ("unsafe-accept", problems) if problems else ("accept", [])


# ------------------------------------------------------------------------------------ integrity
def integrity(workspace: Path, spec: dict) -> tuple[bool, dict]:
    """The pinned entry point, the pools, the registry and the staged case list stay untouched."""
    pinned = json.loads((RESOURCES / "pinned_digests.json").read_text())
    mismatched = [
        rel for rel, expected in sorted(pinned.items())
        if ((workspace / rel).is_file() and digest(workspace / rel) != expected)
        or not (workspace / rel).is_file()
    ]
    pools = {}
    for pool_id, files in sorted(spec["pool_digests"].items()):
        pools[pool_id] = all(
            (workspace / "pools" / pool_id / name).is_file()
            and digest(workspace / "pools" / pool_id / name) == expected
            for name, expected in files.items()
        )
    ok = not mismatched and all(pools.values())
    return ok, {"pinned_files": len(pinned), "mismatched": mismatched[:20],
                "pools_intact": sum(pools.values()), "pools_total": len(pools),
                "pools_modified": sorted(k for k, v in pools.items() if not v)[:10]}


# ----------------------------------------------------------------------------------- hidden pass
def run_hidden_pass(workspace: Path, spec: dict, out_root: Path) -> tuple[list[dict], dict, Path]:
    cases = out_root / "hidden_cases.jsonl"
    shutil.copy2(RESOURCES / "hidden_cases.jsonl", cases)
    logs = out_root / "hidden_logs"
    command = [
        sys.executable, "-B", str(workspace / "scripts" / "run_cases.py"),
        "--cases", str(cases), "--out", str(logs),
        "--timeout-s", str(spec["case_timeout_s"]),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("RUN_TAG", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.time()
    completed = subprocess.run(
        command, cwd=str(workspace), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=spec["hidden_pass_timeout_s"],
    )
    report = {"exit_code": completed.returncode, "seconds": round(time.time() - started, 1),
              "tail": completed.stdout[-1500:]}
    return read_jsonl(logs / "cases.jsonl"), report, cases


# ---------------------------------------------------------------------------------------- score
def score(workspace: Path, records: list[dict], cases: list[dict], truth: dict) -> dict:
    by_case = {case["case_id"]: case for case in cases}
    measured = {record["case_id"]: record for record in records}
    agreed, unsafe, false_accept, false_reject = 0, [], [], []
    detail = {}
    for case_id, expected in sorted(truth.items()):
        record = measured.get(case_id)
        if record is None:
            detail[case_id] = {"outcome": "missing", "expected": expected}
            continue
        facts = pool_facts(workspace, by_case[case_id]["pool_dir"])
        got, problems = outcome(record, facts, by_case[case_id])
        detail[case_id] = {"outcome": got, "expected": expected, "problems": problems[:3]}
        if got == expected:
            agreed += 1
        if got == "unsafe-accept":
            unsafe.append(case_id)
        if expected == "reject" and got != "reject":
            false_accept.append(case_id)
        if expected == "accept" and got == "reject":
            false_reject.append(case_id)
    return {
        "cases": len(truth),
        "agreement": round(agreed / len(truth), 4) if truth else 0.0,
        "unsafe_accepts": len(unsafe),
        "unsafe_examples": unsafe[:10],
        "false_accepts": len(false_accept),
        "false_accept_examples": false_accept[:10],
        "false_rejects": len(false_reject),
        "false_reject_examples": false_reject[:10],
        "missing": sum(1 for value in detail.values() if value["outcome"] == "missing"),
        "detail": detail,
    }


def main() -> int:
    workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
    results = Path(os.environ.get("AUTORLBENCH_RESULTS", "/logs/artifacts"))
    spec = json.loads((RESOURCES / "grader_spec.json").read_text())
    report: dict = {"task": "query_pool_gate_divergence"}

    integrity_ok, integrity_detail = integrity(workspace, spec)
    report["integrity"] = {"passed": integrity_ok, **integrity_detail}

    staged_records = read_jsonl(results / "experiment_logs" / "cases.jsonl")
    staged_cases = read_jsonl(workspace / "cases" / "cases.jsonl")
    staged_truth = json.loads((RESOURCES / "truth_staged.json").read_text())
    report["staged_pass"] = score(workspace, staged_records, staged_cases, staged_truth)

    hidden_score: dict = {}
    hidden_report: dict = {}
    if integrity_ok:
        with tempfile.TemporaryDirectory(prefix="hidden-cases-") as tmp:
            try:
                hidden_records, hidden_report, hidden_cases_path = run_hidden_pass(workspace, spec, Path(tmp))
                hidden_cases = read_jsonl(hidden_cases_path)
            except subprocess.TimeoutExpired:
                hidden_records, hidden_report, hidden_cases = [], {"timed_out": True}, []
            hidden_truth = json.loads((RESOURCES / "truth_hidden.json").read_text())
            hidden_score = score(workspace, hidden_records, hidden_cases, hidden_truth)
    report["hidden_pass"] = {"run": hidden_report, **hidden_score}

    checks = {"integrity": integrity_ok}
    for name, measured in (("staged", report["staged_pass"]), ("hidden", hidden_score)):
        threshold = spec["thresholds"][name]
        checks[f"{name}_agreement"] = bool(measured) and measured["agreement"] >= threshold["agreement"]
        checks[f"{name}_no_unsafe_accept"] = bool(measured) and measured["unsafe_accepts"] == 0
        checks[f"{name}_no_false_accept"] = bool(measured) and measured["false_accepts"] <= threshold["false_accepts"]
        checks[f"{name}_no_false_reject"] = bool(measured) and measured["false_rejects"] <= threshold["false_rejects"]
    report["checks"] = checks
    report["passed"] = all(checks.values())

    results.mkdir(parents=True, exist_ok=True)
    (results / "grading_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
