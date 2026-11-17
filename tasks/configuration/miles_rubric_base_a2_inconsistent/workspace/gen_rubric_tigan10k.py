"""Minimal stub of gen_rubric_tigan10k.py showing the BASE_A2 inconsistency defect.

The real script reads BASE_A2 at TWO distinct call sites:
1. During generation (to populate the a2 answer slot)
2. During --build (to write reward_model.reference into the parquet)

When --build is called without BASE_A2, load_answers() falls back to the
module-level default A2_PARQUET (old base model answers), producing a parquet
where rubric criteria are calibrated against 9B answers but
reward_model.reference contains old-base answers.
"""
import argparse
import json
import os
import sys

# Module-level default: OLD BASE MODEL answers (the bug)
# When --build is called without BASE_A2, this is used instead of the 9B answers.
A2_PARQUET = "data/answers_moebase.parquet"


def load_answers(path=None):
    """Load a2 answers from parquet.

    BUG: When called without BASE_A2 at --build time, this silently reads
    the module-level default A2_PARQUET (old base model answers) instead of
    the 9B answers that were used during generation.

    The fix: always pass BASE_A2 explicitly to both invocations:
        # At generation time:
        BASE_A2=/path/to/answers_q359b.parquet python3 gen_rubric_tigan10k.py
        # At build time:
        BASE_A2=/path/to/answers_q359b.parquet python3 gen_rubric_tigan10k.py --build
    """
    effective_path = os.environ.get("BASE_A2", path or A2_PARQUET)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    effective_path = os.path.join(base_dir, effective_path)
    print(f"[gen_rubric] load_answers reading: {effective_path}")
    # Simulate loading answers
    return {"uid_0000": "答案内容来自: " + effective_path}


def build():
    """Build training parquet from rubric JSONL and reference answers.

    BUG: If BASE_A2 is not set at build time, load_answers() reads the
    module-level default (old base model), producing a parquet where:
    - rubric criteria were calibrated against 9B answers (from generation time)
    - reward_model.reference contains old-base answers (from build time)

    Training proceeds with no error; the mismatch is only detectable by
    sampling 50+ uid pairs and comparing criterion text to reference text.
    """
    answers = load_answers()  # <- BUG: reads default A2_PARQUET if BASE_A2 not set
    rubric_path = os.path.join(os.path.dirname(__file__), "data", "rubric.jsonl")
    rows = []
    with open(rubric_path) as f:
        for line in f:
            row = json.loads(line.strip())
            uid = row["uid"]
            row["reward_model"] = {"reference": answers.get(uid, "MISSING")}
            rows.append(row)
    print(f"[gen_rubric] Built {len(rows)} rows; reference source: {os.environ.get('BASE_A2', A2_PARQUET)}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        rows = build()
        print(f"[gen_rubric] Parquet would be written with reference from: "
              f"{os.environ.get('BASE_A2', A2_PARQUET)}")
    else:
        base_a2 = os.environ.get("BASE_A2", A2_PARQUET)
        print(f"[gen_rubric] Generation mode; BASE_A2={base_a2}")
