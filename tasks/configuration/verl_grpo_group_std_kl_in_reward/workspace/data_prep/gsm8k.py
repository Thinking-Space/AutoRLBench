"""Build verl-format GSM8K parquet files from the raw Hugging Face snapshot.

Adapted from verl examples/data_preprocess/gsm8k.py (Apache-2.0); reads the local
snapshot with pandas so the run needs no network.
"""
import argparse
import os
import re

import pandas as pd

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(answer):
    m = re.search("#### (\\-?[0-9\\.\\,]+)", answer)
    assert m is not None
    return m.group(0).split("#### ")[1].replace(",", "")


def convert(df, split):
    rows = []
    for idx, ex in enumerate(df.to_dict("records")):
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": ex["question"] + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(ex["answer"])},
                "extra_info": {"split": split, "index": idx, "answer": ex["answer"], "question": ex["question"]},
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw snapshot dir containing main/*.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "test"):
        df = pd.read_parquet(os.path.join(a.src, "main", f"{split}-00000-of-00001.parquet"))
        out = convert(df, split)
        out.to_parquet(os.path.join(a.out, f"{split}.parquet"))
        print(f"{split}: {len(out)} rows -> {a.out}/{split}.parquet")
