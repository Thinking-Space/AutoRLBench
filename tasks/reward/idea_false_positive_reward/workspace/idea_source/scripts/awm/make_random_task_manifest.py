#!/usr/bin/env python3
"""Create a reproducible, type-balanced random AWM task manifest."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="/workspace/data/AgentWorldModel-1K/gen_tasks.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--rollouts", type=int, default=1)
    args = p.parse_args()

    rows = [json.loads(line) for line in Path(args.tasks).open(encoding="utf-8") if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scenario = str(row["scenario"])
        # The prefix before the final numeric suffix is the generated task type.
        type_name = scenario.rsplit("_", 1)[0] if scenario.rsplit("_", 1)[-1].isdigit() else scenario
        for task_idx, task in enumerate(row.get("tasks", [])):
            groups[type_name].append({"scenario": scenario, "task_idx": task_idx, "task": task})

    rng = random.Random(args.seed)
    for values in groups.values():
        rng.shuffle(values)
    types = sorted(groups)
    if args.count > sum(len(v) for v in groups.values()):
        raise ValueError("requested count exceeds available tasks")

    # Round-robin over types gives every type representation, then fill the remainder randomly.
    chosen: list[dict] = []
    while len(chosen) < args.count:
        available = [name for name in types if groups[name]]
        if not available:
            break
        rng.shuffle(available)
        for name in available:
            if len(chosen) >= args.count:
                break
            chosen.append(groups[name].pop())
    rng.shuffle(chosen)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in chosen:
            for rollout_idx in range(args.rollouts):
                task_id = f"{item['scenario']}__task_{item['task_idx']}__rollout_{rollout_idx}"
                f.write(json.dumps({**item, "task_id": task_id, "rollout_idx": rollout_idx}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "count": len(chosen), "records": len(chosen) * args.rollouts, "types": len(types), "seed": args.seed}))


if __name__ == "__main__":
    main()
