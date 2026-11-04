#!/usr/bin/env python3
"""Create one deterministic random task manifest shared by all evaluations."""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="/workspace/data/AgentWorldModel-1K/gen_tasks.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    flattened = []
    with Path(args.tasks).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for task_idx, task in enumerate(row.get("tasks", [])):
                flattened.append({
                    "task_id": f"{row['scenario']}__task_{task_idx}__rollout_0",
                    "scenario": row["scenario"],
                    "task_idx": task_idx,
                    "rollout_idx": 0,
                    "task": task,
                })
    if args.count > len(flattened):
        raise ValueError(f"Requested {args.count}, only {len(flattened)} tasks available")
    selected = random.Random(args.seed).sample(flattened, args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "count": len(selected), "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
