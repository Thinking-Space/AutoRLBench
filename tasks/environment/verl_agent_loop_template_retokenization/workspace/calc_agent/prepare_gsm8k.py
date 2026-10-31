"""Convert the raw openai/gsm8k parquet files into the RL dataset used by run_calc_grpo.sh.

  python3 calc_agent/prepare_gsm8k.py --raw data/raw/gsm8k --out data/gsm8k

Writes train.parquet (all 7,473 train problems), test.parquet (512 GSM8K test problems,
fixed-seed sample) and test_hard.parquet (512 GSM-Hard problems: the GSM8K test questions with
large numbers, fixed-seed sample) in verl's RLHFDataset format.

  python3 calc_agent/prepare_gsm8k.py --raw data/raw/gsm8k --raw-hard data/raw/gsm-hard --out data/gsm8k
"""

import argparse
import os
import re

import pandas as pd

SYSTEM_PROMPT = (
    "You are a careful math assistant. You can call the `calculator` tool to evaluate "
    "arithmetic expressions; use it for any non-trivial arithmetic. When you are done, "
    "write the final answer as \\boxed{<number>}."
)
N_TEST = 512
SEED = 2026


def ground_truth(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    assert match is not None, answer
    return match.group(1).replace(",", "")


def convert(df: pd.DataFrame, split: str, data_source: str = "openai/gsm8k") -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(df.itertuples(index=False)):
        rows.append(
            {
                "data_source": data_source,
                "agent_name": "calc_agent",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row.question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": row.answer if split == "test_hard" else ground_truth(row.answer)},
                "extra_info": {"split": split, "index": idx},
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="directory holding main/{train,test}-00000-of-00001.parquet")
    parser.add_argument("--raw-hard", required=True, help="directory holding gsmhardv2.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train = pd.read_parquet(os.path.join(args.raw, "main", "train-00000-of-00001.parquet"))
    test = pd.read_parquet(os.path.join(args.raw, "main", "test-00000-of-00001.parquet"))
    test = test.sample(n=N_TEST, random_state=SEED).reset_index(drop=True)

    os.makedirs(args.out, exist_ok=True)
    convert(train, "train").to_parquet(os.path.join(args.out, "train.parquet"))
    convert(test, "test").to_parquet(os.path.join(args.out, "test.parquet"))

    hard = pd.read_json(os.path.join(args.raw_hard, "gsmhardv2.jsonl"), lines=True)
    hard = hard.sample(n=N_TEST, random_state=SEED).reset_index(drop=True)
    hard = pd.DataFrame({"question": hard["input"], "answer": [f"{float(t):.10g}" for t in hard["target"]]})
    convert(hard, "test_hard", data_source="reasoning-machines/gsm-hard").to_parquet(
        os.path.join(args.out, "test_hard.parquet")
    )
    print(f"train={len(train)} test={len(test)} test_hard={len(hard)} -> {args.out}")


if __name__ == "__main__":
    main()
