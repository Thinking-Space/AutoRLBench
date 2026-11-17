"""Minimal stub of gen_rubric_tigan10k.py showing the premature-build defect.

The real script has async generation writing to a JSONL file.
When --build is called before generation is complete, load_done() reads
whatever rows are present at that moment and builds the parquet immediately.
"""
import argparse
import json
import os
import sys

JSONL_PATH = os.path.join(os.path.dirname(__file__), "data", "rubric_partial.jsonl")
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "data", "train_premature.parquet")


def load_done():
    """Read completed rubric rows from the JSONL.

    BUG: this reads whatever is present at call time.
    It does NOT check whether generation is complete.
    When called while generation is still running, it will read a partial state.

    The correct usage is:
        1. Run python3 gen_rubric_tigan10k.py (no --build) until it prints "待生成 0"
        2. Only then run python3 gen_rubric_tigan10k.py --build
    """
    rows = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_pending():
    """Check how many rows are still pending generation.

    Prints "待生成 0" when generation is complete.
    If this prints a non-zero number, --build must NOT be called yet.
    """
    rows = load_done()
    pending = max(0, 9716 - len(rows))  # 9716 = total expected rows
    print(f"待生成 {pending}")
    return pending


def build():
    """Build the training parquet from completed rubric rows.

    BUG: Called while generation was still at 8199/9716 rows.
    Result: parquet has 8199 rows instead of 9716 (15.6% missing).
    """
    rows = load_done()
    print(f"Building parquet from {len(rows)} rows...")
    # Simulate parquet creation
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({"uid": [r["uid"] for r in rows],
                          "prompt": [r.get("prompt", "") for r in rows]})
        pq.write_table(table, PARQUET_PATH)
        print(f"Wrote {len(rows)} rows to {PARQUET_PATH}")
    except ImportError:
        # Fallback: write a marker file
        with open(PARQUET_PATH, "w") as f:
            f.write(f"PARQUET_ROWS={len(rows)}\n")
        print(f"Wrote {len(rows)}-row marker to {PARQUET_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the parquet")
    args = parser.parse_args()

    if args.build:
        build()
    else:
        pending = check_pending()
        sys.exit(0 if pending == 0 else 1)
