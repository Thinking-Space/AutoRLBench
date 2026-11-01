#!/usr/bin/env python3
"""Create deterministic tiny GSM8K train/validation parquets for a short GRPO run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import datasets

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def extract_solution(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.,]+)", answer)
    if match is None:
        raise ValueError(f"missing GSM8K final answer: {answer!r}")
    return match.group(1).replace(",", "")


def convert(example: dict, index: int, split: str) -> dict:
    question = example["question"]
    answer = example["answer"]
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "system",
                "content": (
                    "Solve the arithmetic problem silently. Your entire response must be exactly "
                    "#### <integer>, with no explanation. Example: #### 5"
                ),
            },
            {
                "role": "user",
                "content": question + " Return only #### followed by the integer answer.",
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": extract_solution(answer)},
        "extra_info": {
            "split": split,
            "index": index,
            "source_index": example["source_index"],
            "selection_rule": "fewest_gsm8k_calculator_steps_then_shortest_question",
            "question": question,
            "answer": answer,
        },
    }


def select_calibration_subset(split: datasets.Dataset, size: int) -> datasets.Dataset:
    source_indices = sorted(
        range(len(split)),
        key=lambda idx: (split[idx]["answer"].count("<<"), len(split[idx]["question"]), idx),
    )[:size]
    return split.select(source_indices).add_column("source_index", source_indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()

    raw = datasets.load_dataset(
        "parquet",
        data_files={
            "train": str(args.source / "main/train-00000-of-00001.parquet"),
            "test": str(args.source / "main/test-00000-of-00001.parquet"),
        },
    )
    train = select_calibration_subset(raw["train"], args.train_size)
    val = select_calibration_subset(raw["test"], args.val_size)
    train = train.map(lambda row, idx: convert(row, idx, "train"), with_indices=True, remove_columns=train.column_names)
    val = val.map(lambda row, idx: convert(row, idx, "test"), with_indices=True, remove_columns=val.column_names)

    args.output.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output / "train.parquet")
    val.to_parquet(args.output / "validation.parquet")
    print(f"tiny GSM8K: train={len(train)} validation={len(val)} output={args.output}")


if __name__ == "__main__":
    main()
