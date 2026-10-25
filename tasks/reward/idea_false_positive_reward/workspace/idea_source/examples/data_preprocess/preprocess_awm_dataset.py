#!/usr/bin/env python3
"""Convert AgentWorldModel-1K tasks to the parquet format used by SEED."""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            scenario = row["scenario"]
            for task_idx, task in enumerate(row.get("tasks", [])):
                tasks.append({"scenario": scenario, "task_idx": task_idx, "task": task})
    return tasks


def to_record(item: dict, split: str, index: int) -> dict:
    return {
        "data_source": "agent_world_model",
        # The environment manager constructs the actual MCP system/task prompt.
        "prompt": [{"role": "user", "content": ""}],
        "ability": "agent",
        "extra_info": {"split": split, "index": index},
        "env_kwargs": item,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/data/AgentWorldModel-1K")
    parser.add_argument("--local-dir", default="/workspace/data/AgentWorldModel-1K/verl")
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.data_root) / "gen_tasks.jsonl")
    if not 0 < args.val_size < len(tasks):
        raise ValueError(f"--val-size must be between 1 and {len(tasks) - 1}")
    random.Random(args.seed).shuffle(tasks)
    splits = {"test": tasks[: args.val_size], "train": tasks[args.val_size :]}

    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        records = [to_record(item, split, idx) for idx, item in enumerate(rows)]
        path = output_dir / f"{split}.parquet"
        Dataset.from_list(records).to_parquet(str(path))
        print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
