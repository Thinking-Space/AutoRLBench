#!/usr/bin/env python3
"""Stage 1 (author-side): the dirty-pool candidate list.

The dirty pool is the set difference BAAI/TACO ALL/train minus
likaixin/TACO-verified: train problems for which NO bundled solution passes all
of the problem's own tests within the verified set's 10 s budget. That pool
mixes three failure families (wrong expected outputs, all-solutions-timeout,
environment mismatch); stages 2/3 (purify) narrow it to the first family.

This script only reads and marshals data (pyarrow, no execution of any bundled
solution), so it runs on the host. It writes one JSON file per candidate
problem -- question, tests, and bundled solutions included -- under --out,
plus a summary dirty_pool.jsonl index. The per-problem files are what the
dockerized purify stage consumes; keeping them separate keeps the execution
stage free of pyarrow and of any dataset parsing it could get wrong.

Usage:
    python3 build_dirty_pool.py \
        --sources <dir with TACO train shards + taco_verified.json> \
        --out <workdir>
    python3 build_dirty_pool.py \
        --sources <dir with TACO test shard> --out <workdir> --test-pool

--test-pool skips the verified-set difference (the verified subset covers
train rows only) and marshals the held-out-split candidates instead, under
problems_test/ + test_pool.jsonl, keyed by test_index. The purify stage then
keeps the problems whose own solutions pass (verdict passes_loose): the
task's held-out set must be verified-clean by execution, not by assumption.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taco_sources import (  # noqa: E402
    code_type_of,
    iter_taco_test,
    iter_taco_train,
    iter_verified_records,
    n_tests_of,
    verify_id_mapping,
)

# Hard cap on a single solution's source size; the purify stage skips anything
# above it (they are near-universally data blobs or obfuscated dumps, and the
# point of the pool is executable correctness evidence).
MAX_SOLUTION_BYTES = 200_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-pool", action="store_true",
                    help="marshal the test-split candidate pool instead of the "
                         "train dirty pool")
    ap.add_argument("--skip-verification", action="store_true",
                    help="skip the id<->row mapping check (debugging only)")
    args = ap.parse_args()

    sources = Path(args.sources)
    out = Path(args.out)

    if args.test_pool:
        return build_test_pool(sources, out)
    return build_dirty_pool(sources, out, args)


def build_test_pool(sources: Path, out: Path) -> int:
    """Marshal the held-out-split candidates (same filters as the dirty pool,
    minus the verified-set difference -- that set covers train rows only)."""
    probs = out / "problems_test"
    probs.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    per_difficulty = collections.Counter()
    with open(out / "test_pool.jsonl", "w", encoding="utf-8") as idx:
        for row in iter_taco_test(sources):
            code_type = code_type_of(row["input_output"])
            n_tests = n_tests_of(row["input_output"])
            sols = [s for s in row["solutions"] if len(s.encode("utf-8", "replace")) <= MAX_SOLUTION_BYTES]
            entry = {
                "test_index": row["test_index"],
                "name": row["name"],
                "difficulty": row["difficulty"],
                "code_type": code_type,
                "n_tests": n_tests,
                "n_solutions": len(sols),
            }
            stats[f"type_{code_type}"] += 1
            if code_type == "stdin" and n_tests > 0 and sols:
                stats["candidate"] += 1
                per_difficulty[row["difficulty"]] += 1
                record = dict(entry)
                record["question"] = row["question"]
                record["input_output"] = row["input_output"]
                record["solutions"] = sols
                with open(probs / f"{row['test_index']}.json", "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False)
            idx.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("[test-pool] composition:")
    for key in sorted(stats):
        print(f"    {key}: {stats[key]}")
    print("[test-pool] executable candidates by difficulty (stdin, n_tests>0, solutions present):")
    for key in sorted(per_difficulty):
        print(f"    {key}: {per_difficulty[key]}")
    return 0


def build_dirty_pool(sources: Path, out: Path, args) -> int:
    probs = out / "problems"
    probs.mkdir(parents=True, exist_ok=True)

    verified_json = sources / "taco_verified.json"
    if not verified_json.is_file():
        print(f"error: {verified_json} missing", file=sys.stderr)
        return 1

    if not args.skip_verification:
        for line in verify_id_mapping(sources, verified_json):
            print("[verify] " + line)

    verified_ids = set()
    for rec in iter_verified_records(verified_json):
        verified_ids.add(int(rec["id"]))
    print(f"[pool] verified problems: {len(verified_ids)}")

    stats = collections.Counter()
    per_difficulty = collections.Counter()
    with open(out / "dirty_pool.jsonl", "w", encoding="utf-8") as idx:
        for row in iter_taco_train(sources):
            # verified ids equal taco_index (0-based concatenated order),
            # verified by verify_id_mapping above
            if row["taco_index"] in verified_ids:
                stats["clean_kept_out"] += 1
                continue
            code_type = code_type_of(row["input_output"])
            n_tests = n_tests_of(row["input_output"])
            sols = [s for s in row["solutions"] if len(s.encode("utf-8", "replace")) <= MAX_SOLUTION_BYTES]
            entry = {
                "taco_index": row["taco_index"],
                "name": row["name"],
                "difficulty": row["difficulty"],
                "code_type": code_type,
                "n_tests": n_tests,
                "n_solutions": len(sols),
            }
            stats[f"type_{code_type}"] += 1
            if code_type == "stdin" and n_tests > 0 and sols:
                stats["candidate"] += 1
                per_difficulty[row["difficulty"]] += 1
                record = dict(entry)
                record["question"] = row["question"]
                record["input_output"] = row["input_output"]
                record["solutions"] = sols
                with open(probs / f"{row['taco_index']}.json", "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False)
            idx.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("[pool] dirty-pool composition:")
    for key in sorted(stats):
        print(f"    {key}: {stats[key]}")
    print("[pool] executable candidates by difficulty (stdin, n_tests>0, solutions present):")
    for key in sorted(per_difficulty):
        print(f"    {key}: {per_difficulty[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
