"""Re-grade the staged rollout corpus with the workspace's grading pipeline.

This is the harness entry point: it reads the corpus manifest, grades every record with
`rlgrade.grade_record`, and writes one JSON line per record to
`$AUTORLBENCH_RESULTS/experiment_logs/regrade.jsonl` plus a run summary. It owns no grading
logic of its own -- every decision it records comes out of the `rlgrade` package.

usage: python3 -B scripts/regrade_run.py [--manifest PATH] [--out DIR] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from rlgrade import grade_record  # noqa: E402
from rlgrade.types import Instance, Submission  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def results_root() -> Path:
    root = os.environ.get("AUTORLBENCH_RESULTS")
    return Path(root) if root else WORKSPACE


def load_config() -> dict:
    """Knobs of the grading pass: scripts/run_config.json, overridable on the command line."""
    path = WORKSPACE / "scripts" / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return config


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(WORKSPACE / config.get("manifest", "corpus/manifest.jsonl")))
    parser.add_argument("--records", default=str(WORKSPACE / config.get("records", "corpus/records.jsonl")))
    parser.add_argument("--out", default=None, help="log directory (default: $AUTORLBENCH_RESULTS/experiment_logs)")
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--keep-checkouts", action="store_true",
                        help="keep each record's scratch checkout after grading it")
    parser.add_argument("--workers", type=int, default=int(config.get("workers", 4)))
    parser.add_argument("--timeout-s", type=int, default=int(config.get("timeout_s", 900)))
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    corpus_root = manifest_path.parent
    instances = {
        row["instance_id"]: Instance.from_manifest(row, corpus_root)
        for row in read_jsonl(manifest_path)
    }
    records = read_jsonl(Path(args.records).resolve())

    out_dir = Path(args.out) if args.out else results_root() / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch) if args.scratch else out_dir / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    lines: list[str] = []

    def one(row: dict) -> dict:
        submission = Submission(
            record_id=row["record_id"],
            instance_id=row["instance_id"],
            patch=(corpus_root / row["patch"]).read_text(encoding="utf-8") if row.get("patch") else "",
            metadata=row.get("metadata", {}),
        )
        instance = instances[row["instance_id"]]
        scratch = scratch_root / row["record_id"]
        t0 = time.time()
        try:
            grade = grade_record(instance, submission, scratch, timeout_s=args.timeout_s)
            payload = grade.to_json()
        except Exception as exc:  # a crashed record must not take the whole pass down
            payload = {
                "record_id": row["record_id"],
                "instance_id": row["instance_id"],
                "reward": 0.0,
                "label": "error",
                "missing": [],
                "failed": [],
                "notes": [f"{type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc()[-2000:],
            }
        payload["seconds"] = round(time.time() - t0, 2)
        if not args.keep_checkouts:
            # The raw pytest log stays in the scratch directory; the checkout does not.
            shutil.rmtree(scratch / instance.repo_key, ignore_errors=True)
        return payload

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for payload in pool.map(one, records):
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            print(f"{payload['record_id']} label={payload['label']} reward={payload['reward']}", flush=True)

    (out_dir / "regrade.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    graded = [json.loads(line) for line in lines]
    scored = [row for row in graded if isinstance(row.get("reward"), (int, float))]
    summary = {
        "records": len(graded),
        "scored": len(scored),
        "rewarded": sum(1 for row in scored if row["reward"] >= 1.0),
        "labels": {label: sum(1 for row in graded if row["label"] == label) for label in sorted({row["label"] for row in graded})},
        "mean_reward": round(sum(row["reward"] for row in scored) / max(1, len(scored)), 4),
        "seconds": round(time.time() - started, 1),
        "pipeline": "rlgrade",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
