"""Build a task-level full-cloud cost baseline from forced-cloud rollout dumps."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def build_baseline(dump_dir: Path, output: Path) -> dict[str, Any]:
    paths = sorted(path for path in dump_dir.glob("rollout_*.pt") if not path.name.startswith("rollout_eval_"))
    if not paths:
        raise ValueError(f"no rollout_*.pt files found under {dump_dir}")

    rollouts: dict[tuple[str, Any], dict[str, Any]] = {}
    cloud_model = None
    pricing = None
    advisor_style = None
    for path in paths:
        payload = torch.load(path, weights_only=False)
        for sample in payload.get("samples", []):
            metadata = sample.get("metadata") or {}
            if not metadata.get("advisor_force_cloud"):
                raise ValueError(f"{path} contains a sample that was not generated with force-cloud calibration")
            task_id = str(metadata.get("task_id") or "")
            if not task_id:
                raise ValueError(f"{path} contains a sample without task_id")
            calls = int(metadata.get("advisor_calls", 0) or 0)
            cost = float(metadata.get("advisor_cloud_cost_rmb", 0.0) or 0.0)
            if calls < 1 or not math.isfinite(cost) or cost <= 0:
                raise ValueError(
                    f"forced-cloud rollout for {task_id} has invalid calls/cost: calls={calls}, cost={cost}"
                )
            model = metadata.get("advisor_cloud_model")
            sample_pricing = metadata.get("advisor_cost_pricing_rmb_per_1k")
            style = str(metadata.get("advisor_style") or "").lower()
            if cloud_model is None:
                cloud_model, pricing, advisor_style = model, sample_pricing, style
            if model != cloud_model or not _same(sample_pricing, pricing) or style != advisor_style:
                raise ValueError("rollout dumps mix cloud models, advisor styles, or pricing configurations")
            rollout_key = sample.get("rollout_id") if sample.get("rollout_id") is not None else sample.get("index")
            key = (task_id, rollout_key)
            record = {"cost_rmb": cost, "calls": calls}
            if key in rollouts and not _same(rollouts[key], record):
                raise ValueError(f"fan-out samples disagree for rollout {key!r}")
            rollouts[key] = record

    costs_by_task: dict[str, list[float]] = defaultdict(list)
    calls_by_task: dict[str, list[int]] = defaultdict(list)
    for (task_id, _rollout_id), record in rollouts.items():
        costs_by_task[task_id].append(record["cost_rmb"])
        calls_by_task[task_id].append(record["calls"])
    task_costs = {task_id: sum(costs) / len(costs) for task_id, costs in sorted(costs_by_task.items())}
    result = {
        "version": "openclaw-full-cloud-cost-v1",
        "cloud_model": cloud_model,
        "advisor_style": advisor_style,
        "pricing_rmb_per_1k": pricing,
        "task_cost_rmb": task_costs,
        "samples_per_task": {task_id: len(costs) for task_id, costs in sorted(costs_by_task.items())},
        "mean_calls_per_task": {task_id: sum(calls) / len(calls) for task_id, calls in sorted(calls_by_task.items())},
        "num_tasks": len(task_costs),
        "num_rollouts": len(rollouts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dumps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_baseline(args.rollout_dumps, args.output)
    print(f"wrote {result['num_tasks']} tasks from {result['num_rollouts']} rollouts to {args.output}")


if __name__ == "__main__":
    main()
