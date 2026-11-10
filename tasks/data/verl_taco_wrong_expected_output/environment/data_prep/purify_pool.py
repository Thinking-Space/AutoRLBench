#!/usr/bin/env python3
"""Stage 2 (author-side, in-container): purify the dirty pool by re-execution.

Drives exec_problem.py over the candidate problems produced by
build_dirty_pool.py, in a seeded random order, with N worker subprocesses in
parallel. Every problem's verdict is appended to --receipts as soon as it
finishes, so the stage is resumable: problems already present in the receipts
file are skipped on restart.

Termination:
  * --max-problems N      stop after N problems tried (budget);
  * --target-dirty N      stop early once N problems have the
                          native_wrong_expected_output verdict;
  * full pool exhausted.
On any of these, or on Ctrl-C, the dirty manifest and the corrected-outputs
table are (re)written from whatever receipts exist.

Outputs:
  receipts.jsonl          one line per problem tried (audit trail; keep it)
  dirty_manifest.jsonl    verdict == native_wrong_expected_output, with the
                          failing test index, the dataset's expected output, the
                          agreeing solutions' actual output, and the solution
                          hashes -- the per-problem receipts the task's evidence
                          plan cites.
  corrected_outputs.json  question_sha256 -> {fail_index, fail_input,
                          corrected_output, ...}; the external-judge table the
                          solution grader uses to score "correct but judged 0"
                          rollouts.
  --passing-manifest PATH optional extra output: problems with verdict
                          passes_loose (>= 1 bundled solution passes all tests
                          under the loose budget). Used on the test-split pool
                          to establish held-out cleanliness by execution.

Usage (inside the container, see data_prep/README.md):
    python3 purify_pool.py \
        --problems /work/problems \
        --receipts /work/receipts.jsonl \
        --manifest /work/dirty_manifest.jsonl \
        --corrected /work/corrected_outputs.json \
        --taco /task/workspace/verl/utils/reward_score/taco.py \
        --difficulties EASY,MEDIUM --workers 8
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HARD_WALL_S = 45 * 60  # per-problem wall clock kill


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problems", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--passing-manifest", default="",
                    help="optional: also write the passes_loose problems here "
                         "(test-split pool -> held-out cleanliness receipts)")
    ap.add_argument("--taco", required=True)
    ap.add_argument("--difficulties", default="EASY,MEDIUM")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-solutions", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-problems", type=int, default=0, help="0 = no cap")
    ap.add_argument("--target-dirty", type=int, default=0, help="0 = no cap")
    ap.add_argument("--order-seed", type=int, default=20261106)
    ap.add_argument("--wall-seconds", type=float, default=HARD_WALL_S)
    args = ap.parse_args()

    difficulties = {d.strip().upper() for d in args.difficulties.split(",") if d.strip()}
    problems_dir = Path(args.problems)
    receipts_path = Path(args.receipts)
    receipts_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if receipts_path.exists():
        with open(receipts_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["taco_index"])
                except (ValueError, KeyError):
                    continue
        print(f"[purify] resuming, {len(done)} problems already tried", flush=True)

    files = []
    for p in problems_dir.glob("*.json"):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx in done:
            continue
        files.append(p)
    # Difficulty filter needs the record; read lazily in the order pass.
    rng = random.Random(args.order_seed)
    rng.shuffle(files)

    write_lock = threading.Lock()
    stop = threading.Event()
    counters = collections.Counter()
    started = time.time()

    def run_one(problem_file: Path):
        if stop.is_set():
            return None
        try:
            record = json.loads(problem_file.read_text(encoding="utf-8"))
        except ValueError:
            return {"taco_index": problem_file.stem, "verdict": "bad_record"}
        if str(record.get("difficulty", "")).upper() not in difficulties:
            return None
        cmd = [
            sys.executable, str(Path(__file__).with_name("exec_problem.py")),
            "--problem", str(problem_file),
            "--taco", args.taco,
            "--timeout", str(args.timeout),
            "--max-solutions", str(args.max_solutions),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=args.wall_seconds,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            result = json.loads(line) if line else {
                "taco_index": record.get("taco_index"),
                "verdict": "no_output",
                "stderr": (proc.stderr or "")[-2000:],
            }
        except subprocess.TimeoutExpired:
            result = {
                "taco_index": record.get("taco_index"),
                "name": record.get("name"),
                "difficulty": record.get("difficulty"),
                "verdict": "harness_timeout",
            }
        except Exception as exc:
            result = {
                "taco_index": record.get("taco_index"),
                "verdict": "error",
                "error": repr(exc),
            }
        with write_lock:
            with open(receipts_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            counters[result.get("verdict", "?")] += 1
            counters["_tried"] += 1
            if result.get("verdict") == "native_wrong_expected_output":
                counters.setdefault("_dirty", 0)
                counters["_dirty"] += 1
        return result

    selected = []
    for f in files:
        if args.max_problems and len(selected) >= args.max_problems:
            break
        selected.append(f)

    print(f"[purify] {len(selected)} problems queued "
          f"(difficulty filter {sorted(difficulties)})", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, f) for f in selected]
            for fut in as_completed(futures):
                fut.result()
                if args.target_dirty and counters.get("_dirty", 0) >= args.target_dirty:
                    print(f"[purify] target-dirty {args.target_dirty} reached; "
                          "draining in-flight problems", flush=True)
                    stop.set()
                    break
    except KeyboardInterrupt:
        print("[purify] interrupted; writing manifest from receipts so far", flush=True)

    write_outputs(receipts_path, Path(args.manifest), Path(args.corrected),
                  passing_path=Path(args.passing_manifest) if args.passing_manifest else None)
    print(f"[purify] done in {time.time() - started:.0f}s; verdicts: "
          f"{dict(counters)}", flush=True)
    return 0


def write_outputs(receipts_path: Path, manifest_path: Path, corrected_path: Path,
                   passing_path: Path | None = None) -> None:
    dirty = []
    corrected = {}
    passing = []
    seen_question_sha = set()
    with open(receipts_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("verdict") == "native_wrong_expected_output":
                # Stricter criterion (see exec_problem.py): an empty agreed
                # output is the unparseable-test-input family, not evidence
                # that the expected output is wrong. Enforced here too so
                # receipts produced under the older, looser criterion are
                # re-filtered on resume/rewrite rather than re-executed.
                if not rec.get("corrected_output"):
                    continue
                # TACO carries a few duplicate questions with DIFFERENT test
                # sets; the grader keys corrected outputs by question sha256,
                # so keep the first occurrence only (a later twin would be
                # judged against the wrong failing test).
                qsha = str(rec.get("question_sha256"))
                if qsha in seen_question_sha:
                    continue
                seen_question_sha.add(qsha)
                entry = {
                    "taco_index": rec["taco_index"],
                    "name": rec.get("name"),
                    "difficulty": rec.get("difficulty"),
                    "n_tests": rec.get("n_tests"),
                    "question_sha256": rec.get("question_sha256"),
                    "fail_index": rec.get("fail_index"),
                    "expected": rec.get("expected"),
                    "corrected_output": rec.get("corrected_output"),
                    "agreeing_solutions": rec.get("agreeing_solutions"),
                }
                dirty.append(entry)
                corrected[str(rec.get("question_sha256"))] = {
                    "taco_index": rec["taco_index"],
                    "name": rec.get("name"),
                    "difficulty": rec.get("difficulty"),
                    "fail_index": rec.get("fail_index"),
                    "fail_input": rec.get("fail_input"),
                    "expected": rec.get("expected"),
                    "corrected_output": rec.get("corrected_output"),
                }
            elif rec.get("verdict") == "passes_loose" and passing_path is not None:
                passing.append({
                    "test_index": rec["taco_index"],
                    "name": rec.get("name"),
                    "difficulty": rec.get("difficulty"),
                    "n_tests": rec.get("n_tests"),
                })
    dirty.sort(key=lambda e: e["taco_index"])
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for e in dirty:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(corrected_path, "w", encoding="utf-8") as fh:
        json.dump(corrected, fh, ensure_ascii=False, indent=1)
    if passing_path is not None:
        passing.sort(key=lambda e: e["test_index"])
        with open(passing_path, "w", encoding="utf-8") as fh:
            for e in passing:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[purify] passing manifest: {len(passing)} problems -> "
              f"{passing_path}", flush=True)
    print(f"[purify] manifest: {len(dirty)} native-wrong-expected-output problems -> "
          f"{manifest_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
