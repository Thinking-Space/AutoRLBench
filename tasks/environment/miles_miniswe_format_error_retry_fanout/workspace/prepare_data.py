#!/usr/bin/env python3
"""Select complete SWE-bench Verified records and Miles prompt rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EXCLUDED_INSTANCE_IDS = {
    "astropy__astropy-7606",
    "django__django-10097",
    "scikit-learn__scikit-learn-14710",
    "sphinx-doc__sphinx-8595",
    "sphinx-doc__sphinx-9711",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_round_robin(rows: list[dict], count: int) -> list[dict]:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if str(row["instance_id"]) in EXCLUDED_INSTANCE_IDS:
            continue
        by_repo[str(row["repo"])].append(row)

    selected: list[dict] = []
    cursor = 0
    while len(selected) < count:
        repositories = [repo for repo in by_repo if cursor < len(by_repo[repo])]
        if not repositories:
            break
        for repo in sorted(repositories):
            if len(selected) >= count:
                break
            selected.append(by_repo[repo][cursor])
        cursor += 1
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = list(dataset)
    selected = select_round_robin(rows, args.num_tasks)
    if len(selected) != args.num_tasks:
        raise AssertionError(f"requested {args.num_tasks} tasks, selected {len(selected)}")

    instance_ids = [str(row["instance_id"]) for row in selected]
    if len(set(instance_ids)) != len(instance_ids):
        raise AssertionError("selected instance IDs are not unique")

    prompt_rows = [
        {
            "prompt": row["problem_statement"],
            "metadata": {
                "instance_id": row["instance_id"],
                "agent_name": "mini-swe-agent",
            },
        }
        for row in selected
    ]

    output_dir = args.output_dir
    write_jsonl(output_dir / "swebench_verified.jsonl", selected)
    write_jsonl(output_dir / "miles_prompts.jsonl", prompt_rows)
    (output_dir / "selected_instance_ids.txt").write_text(
        "".join(f"{instance_id}\n" for instance_id in instance_ids),
        encoding="utf-8",
    )
    print(f"selected {len(selected)} SWE-bench Verified instances")


if __name__ == "__main__":
    main()
