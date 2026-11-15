"""Convert openai/gsm8k (main config) into veRL RL parquet files.

Training rows whose question is longer than MAX_QUESTION_CHARS are dropped: there are
few of them and they stretch the padded prompt length of every batch.
"""
import argparse
import os
import re

import pandas as pd

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
MAX_QUESTION_CHARS = 500


def extract_solution(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    assert match is not None, answer
    return match.group(1).replace(",", "")


def load_split(src: str, split: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(os.path.join(src, "main")) if f.startswith(split))
    return pd.concat([pd.read_parquet(os.path.join(src, "main", f)) for f in files], ignore_index=True)


def build_rows(questions: list[str], answers: list[str], split: str) -> pd.DataFrame:
    rows = []
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer)},
                "extra_info": {"split": split, "index": idx, "answer": answer, "question": question},
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="openai/gsm8k snapshot (main/train-*.parquet, main/test-*.parquet)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "test"):
        df = load_split(args.src, split)
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
        if split == "train":
            questions = [q for q in questions if len(q) <= MAX_QUESTION_CHARS]
        rows = build_rows(questions, answers, split)
        rows.to_parquet(os.path.join(args.out, f"{split}.parquet"))
        print(f"{split}: {len(rows)} rows")


if __name__ == "__main__":
    main()
