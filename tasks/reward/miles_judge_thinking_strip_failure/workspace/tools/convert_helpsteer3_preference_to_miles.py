#!/usr/bin/env python3
"""Convert HelpSteer3 preference parquet data to Miles JSONL format.

Input:  /user/lixin/datasets/nvidia/HelpSteer3/preference/{train,test}.parquet
Output: /user/lixin/datasets/miles/helpsteer3_preference_{train,validation}.jsonl

Each output line: {"prompt": [...messages...], "label": "chosen_response", "metadata": {...}}
"""

import argparse
import json
import os

import pyarrow.parquet as pq


def convert_split(input_path: str, output_path: str) -> int:
    table = pq.read_table(input_path)
    data = table.to_pydict()
    n = len(data["prompt"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            prompt = data["prompt"][i]
            rm_info = data["reward_model"][i]
            extra = data["extra_info"][i]

            chosen = rm_info.get("ground_truth", "")
            if not chosen:
                continue

            record = {
                "prompt": prompt,
                "label": chosen,
                "metadata": {
                    "domain": extra.get("domain"),
                    "language": extra.get("language"),
                    "data_source": data["data_source"][i],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return n


def main():
    parser = argparse.ArgumentParser(description="Convert HelpSteer3 preference data to Miles format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/user/lixin/datasets/nvidia/HelpSteer3/preference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/user/lixin/datasets/miles",
    )
    args = parser.parse_args()

    splits = [
        ("train.parquet", "helpsteer3_preference_train.jsonl"),
        ("test.parquet", "helpsteer3_preference_validation.jsonl"),
    ]

    for in_name, out_name in splits:
        in_path = os.path.join(args.input_dir, in_name)
        out_path = os.path.join(args.output_dir, out_name)

        if not os.path.exists(in_path):
            print(f"Skipping {in_path} (not found)")
            continue

        count = convert_split(in_path, out_path)
        print(f"Converted {count} rows: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
