"""Run the launcher over every case and record what the gate decided.

This is the harness entry point. It owns no gate logic: it sets each case's environment, runs
`scripts/launch_run.sh`, and writes one JSON line per case to
`$AUTORLBENCH_RESULTS/experiment_logs/cases.jsonl`, plus a summary.

Per case the record holds the gate's decision (`accepted`), the message it failed with, and --
when the run started -- the run identity and the batches the stub consumer actually assembled.

usage: python3 -B scripts/run_cases.py [--cases PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
LAUNCHER = WORKSPACE / "scripts" / "launch_run.sh"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def results_root() -> Path:
    root = os.environ.get("AUTORLBENCH_RESULTS")
    return Path(root) if root else WORKSPACE


def run_case(case: dict, out_dir: Path, timeout_s: int) -> dict:
    case_log = out_dir / "cases" / case["case_id"]
    if case_log.exists():
        shutil.rmtree(case_log)
    case_log.mkdir(parents=True)

    environment = dict(os.environ)
    environment.pop("RUN_TAG", None)
    environment.update({key: str(value) for key, value in case.get("env", {}).items()})
    environment["RUN_LOG_DIR"] = str(case_log)
    environment["JOB_ID"] = case.get("job_id", "000000")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pool_dir = WORKSPACE / case["pool_dir"]
    environment.setdefault("POOL_DIR", str(pool_dir))
    environment["POOL_DIR"] = str(pool_dir)

    started = time.time()
    try:
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=str(WORKSPACE),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        exit_code, output, timed_out = completed.returncode, completed.stdout, False
    except subprocess.TimeoutExpired as expired:
        exit_code, output, timed_out = -1, (expired.stdout or ""), True

    (case_log / "launcher.log").write_text(output, encoding="utf-8")
    record = {
        "case_id": case["case_id"],
        "pool_dir": case["pool_dir"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "accepted": exit_code == 0,
        "seconds": round(time.time() - started, 2),
        "message": output.strip().splitlines()[-1][:400] if output.strip() else "",
    }
    run_path = case_log / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        record["run"] = {
            "run_tag": run["run_tag"],
            "pool_total_env": run["pool_total_env"],
            "rows_read": run["rows_read"],
            "unique_rows": run["unique_rows"],
            "pool_source_counts": run["pool_source_counts"],
            "rollout_config": run["rollout_config"],
            "duplicate_prompts_total": run["duplicate_prompts_total"],
            "batch_source_mix": run["batches"][0]["source_mix"] if run["batches"] else {},
        }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(WORKSPACE / "cases" / "cases.jsonl"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args(argv)

    cases = read_jsonl(Path(args.cases).resolve())
    out_dir = Path(args.out) if args.out else results_root() / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case in cases:
        record = run_case(case, out_dir, args.timeout_s)
        records.append(record)
        print(f"{record['case_id']} accepted={record['accepted']} exit={record['exit_code']}", flush=True)

    (out_dir / "cases.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "cases": len(records),
        "accepted": sum(1 for record in records if record["accepted"]),
        "rejected": sum(1 for record in records if not record["accepted"]),
        "gate": "scripts/launch_run.sh",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
