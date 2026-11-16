"""Convert google-research-datasets/mbpp (full config) into veRL RL parquet files.

Train = train + validation + prompt splits (474 problems); test = test split (500 problems).
The prompt shows the first assert so the model knows the function name and signature;
the reward runs all asserts.
"""
import argparse
import json
import os

import pandas as pd

TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {text}\n"
    "Your code should pass this test:\n\n{test}\n\n"
    "Write the complete function in a single ```python code block."
)


def load(src: str, split: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(os.path.join(src, "full")) if f.startswith(split))
    return pd.concat([pd.read_parquet(os.path.join(src, "full", f)) for f in files], ignore_index=True)


def build_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for idx, ex in enumerate(df.itertuples(index=False)):
        tests = [str(t) for t in ex.test_list]
        rows.append(
            {
                "data_source": "google-research-datasets/mbpp",
                "prompt": [{"role": "user", "content": TEMPLATE.format(text=ex.text, test=tests[0])}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"setup": ex.test_setup_code or "", "tests": tests}),
                },
                "extra_info": {"split": split, "index": idx, "task_id": int(ex.task_id)},
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train = pd.concat([load(args.src, s) for s in ("train", "validation", "prompt")], ignore_index=True)
    test = load(args.src, "test")
    for split, df in (("train", train), ("test", test)):
        rows = build_rows(df, split)
        rows.to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
