"""Build verl-format parquet files from the raw GSM-Hard snapshot (reasoning-machines/gsm-hard, MIT).

GSM-Hard rewrites GSM8K test problems with large numbers, so exact arithmetic matters. The 1,319 rows
are split deterministically (sha1 of the question text) into ~80% train / ~20% held-out.
"""
import argparse
import hashlib
import json
import os

import pandas as pd

SYSTEM = (
    "You are a careful math assistant. Reason step by step. You can call the `python` tool to run "
    "Python code for any arithmetic; print the values you need. After you have the result, give the "
    "final answer in the format `#### <answer>`."
)
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def split_of(question):
    return "test" if int(hashlib.sha1(question.encode()).hexdigest(), 16) % 5 == 0 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw snapshot dir containing gsmhardv2.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = {"train": [], "test": []}
    with open(os.path.join(a.src, "gsmhardv2.jsonl")) as f:
        for line in f:
            ex = json.loads(line)
            split = split_of(ex["input"])
            rows[split].append(
                {
                    "data_source": "gsm_hard",
                    "agent_name": "tool_agent",
                    "prompt": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["input"] + " " + INSTRUCTION},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": repr(float(ex["target"]))},
                    "extra_info": {"split": split, "index": len(rows[split]), "question": ex["input"]},
                }
            )
    for split, r in rows.items():
        pd.DataFrame(r).to_parquet(os.path.join(a.out, f"{split}.parquet"))
        print(f"{split}: {len(r)} rows -> {a.out}/{split}.parquet")


if __name__ == "__main__":
    main()
