"""
Convert verl-format parquet/HF datasets into miles jsonl format for retool training.

Training data: parquet with columns [data_source, prompt, ability, reward_model, extra_info, pass_rate]
Eval data: AIME_2024 / aime_2025 / aime_2026_I with varying schemas

Output: jsonl with {"prompt": [{"role": "user", "content": "..."}], "label": "...", "metadata": {"data_source": "..."}}
"""

import json
import argparse
from pathlib import Path

import pandas as pd


ANSWER_FORMAT = "\nThe answer format must be: \\boxed{'The final answer goes here.'}"


def convert_train_parquet(src_path: str, dst_path: str):
    df = pd.read_parquet(src_path)
    records = []
    for _, row in df.iterrows():
        prompt_msgs = row["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        if isinstance(prompt_msgs, list) and len(prompt_msgs) > 0:
            content = prompt_msgs[0]["content"]
        else:
            content = str(prompt_msgs)

        gt = row["reward_model"]
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", "")
        label = str(gt)

        records.append({
            "prompt": [{"role": "user", "content": content}],
            "label": label,
            "metadata": {"data_source": row.get("data_source", "unknown")},
        })

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {dst_path}")


def convert_aime_2024(src_path: str, dst_path: str):
    df = pd.read_parquet(src_path)
    records = []
    for _, row in df.iterrows():
        content = row["Problem"] + ANSWER_FORMAT
        records.append({
            "prompt": [{"role": "user", "content": content}],
            "label": str(row["Answer"]),
            "metadata": {"data_source": "aime_2024"},
        })
    with open(dst_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {dst_path}")


def convert_aime_2025(src_dir: str, dst_path: str):
    import glob
    parquets = glob.glob(f"{src_dir}/**/train-*.parquet", recursive=True)
    if not parquets:
        parquets = glob.glob(f"{src_dir}/**/*.parquet", recursive=True)
    records = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            content = row["problem"] + ANSWER_FORMAT
            records.append({
                "prompt": [{"role": "user", "content": content}],
                "label": str(row["answer"]),
                "metadata": {"data_source": "aime_2025"},
            })
    with open(dst_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {dst_path}")


def convert_aime_2026(src_dir: str, dst_path: str):
    import glob
    parquets = glob.glob(f"{src_dir}/**/train-*.parquet", recursive=True)
    if not parquets:
        parquets = glob.glob(f"{src_dir}/**/*.parquet", recursive=True)
    records = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            content = row["problem"] + ANSWER_FORMAT
            records.append({
                "prompt": [{"role": "user", "content": content}],
                "label": str(row["answer"]),
                "metadata": {"data_source": "aime_2026"},
            })
    with open(dst_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/user/luozhenghang/0226_miles/data/retool")
    args = parser.parse_args()

    out = Path(args.output_dir)

    # Training data
    convert_train_parquet(
        "/user/lixin/datasets/rl_pass_rate/combined_dapo_passrate_le075_and_ge00_and_deepscaler_solve_percent_half_train.parquet",
        str(out / "train" / "retool_train.jsonl"),
    )

    # Eval data
    convert_aime_2024(
        "/user/lixin/datasets/AIME_2024/aime_2024_problems.parquet",
        str(out / "eval" / "aime_2024.jsonl"),
    )
    convert_aime_2025(
        "/user/lixin/datasets/aime_2025",
        str(out / "eval" / "aime_2025.jsonl"),
    )
    convert_aime_2026(
        "/user/lixin/datasets/aime_2026_I",
        str(out / "eval" / "aime_2026_I.jsonl"),
    )

    # Merge all eval into one file for convenience
    eval_dir = out / "eval"
    merged = []
    for f in sorted(eval_dir.glob("aime_*.jsonl")):
        if f.name == "aime_all.jsonl":
            continue
        with open(f) as fh:
            merged.extend(fh.readlines())
    with open(eval_dir / "aime_all.jsonl", "w") as fh:
        fh.writelines(merged)
    print(f"Merged {len(merged)} eval records to {eval_dir / 'aime_all.jsonl'}")


if __name__ == "__main__":
    main()
