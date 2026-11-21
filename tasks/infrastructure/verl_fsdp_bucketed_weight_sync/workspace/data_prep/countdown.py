"""Convert Jiayi-Pan/Countdown-Tasks-3to4 into veRL RL parquet files."""
import argparse
import json
import os

import datasets
import pyarrow.parquet as pq

TRAIN_ROWS = 32768
TEST_ROWS = 1024

PROMPT_TEMPLATE = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and parentheses, "
    "and each number must be used once. Show your reasoning and output the "
    "final expression (without '= {target}') inside <answer>...</answer> tags, "
    "for example <answer>(1 + 2) * 3</answer>."
)


def build_split(targets: list, nums_list: list, indices: list[int], split: str) -> datasets.Dataset:
    rows = []
    for idx in indices:
        target = int(targets[idx])
        numbers = [int(x) for x in nums_list[idx]]
        content = PROMPT_TEMPLATE.format(nums=numbers, target=target)
        rows.append(
            {
                "data_source": "Jiayi-Pan/Countdown-Tasks-3to4",
                "prompt": [{"role": "user", "content": content}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"numbers": numbers, "target": target}, sort_keys=True),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "target": target,
                    "numbers": json.dumps(numbers),
                },
            }
        )
    return datasets.Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Jiayi-Pan/Countdown-Tasks-3to4 snapshot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = pq.read_table(os.path.join(args.src, "data", "train-00000-of-00001.parquet"))
    total = table.num_rows
    assert total >= TRAIN_ROWS + TEST_ROWS, f"Need >= {TRAIN_ROWS + TEST_ROWS} rows, got {total}"
    targets = table["target"].to_pylist()
    nums_list = table["nums"].to_pylist()
    os.makedirs(args.out, exist_ok=True)
    train_ds = build_split(targets, nums_list, list(range(TRAIN_ROWS)), "train")
    test_ds = build_split(targets, nums_list, list(range(total - TEST_ROWS, total)), "test")
    train_ds.to_parquet(os.path.join(args.out, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.out, "test.parquet"))
    print(f"train: {len(train_ds)} rows, test: {len(test_ds)} rows (disjoint from {total} total)")


if __name__ == "__main__":
    main()
