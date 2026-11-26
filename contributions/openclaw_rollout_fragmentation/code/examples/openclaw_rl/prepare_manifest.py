from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from examples.openclaw_rl.tasks import discover_tasks


FULL_CLOUD_METADATA_KEYS = (
    "full_cloud_cost_rmb",
    "full_cloud_cost_model",
    "full_cloud_cost_advisor_style",
    "full_cloud_cost_pricing_rmb_per_1k",
    "full_cloud_cost_source",
    "full_cloud_cost_baseline_version",
)


def _load_cost_metadata_from_baseline(path: Path) -> dict[str, dict[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if payload.get("version") != "openclaw-full-cloud-cost-v1":
        raise ValueError(f"unsupported full-cloud cost baseline version in {path}")
    raw_costs = payload.get("task_cost_rmb")
    if not isinstance(raw_costs, dict) or not raw_costs:
        raise ValueError("full-cloud cost baseline must contain a non-empty task_cost_rmb object")
    common = {
        "full_cloud_cost_model": str(payload.get("cloud_model") or ""),
        "full_cloud_cost_advisor_style": str(payload.get("advisor_style") or "").lower(),
        "full_cloud_cost_pricing_rmb_per_1k": payload.get("pricing_rmb_per_1k"),
        "full_cloud_cost_source": str(payload.get("source") or ""),
        "full_cloud_cost_baseline_version": str(payload["version"]),
    }
    if not common["full_cloud_cost_model"] or not common["full_cloud_cost_advisor_style"]:
        raise ValueError("full-cloud cost baseline must identify cloud_model and advisor_style")
    if not isinstance(common["full_cloud_cost_pricing_rmb_per_1k"], dict):
        raise ValueError("full-cloud cost baseline must contain pricing_rmb_per_1k")
    result: dict[str, dict[str, Any]] = {}
    for task_id, raw_cost in raw_costs.items():
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"full-cloud cost for {task_id!r} must be finite and positive")
        result[str(task_id)] = {"full_cloud_cost_rmb": cost, **common}
    return result


def _load_cost_metadata_from_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            task_id = str(row.get("label") or "")
            metadata = row.get("metadata") or {}
            if not task_id or "full_cloud_cost_rmb" not in metadata:
                continue
            missing = [key for key in FULL_CLOUD_METADATA_KEYS if key not in metadata]
            if missing:
                raise ValueError(f"manifest line {line_number} has incomplete full-cloud metadata: {missing}")
            result[task_id] = {key: metadata[key] for key in FULL_CLOUD_METADATA_KEYS}
    return result


def build_manifest(
    dataset_dir: Path,
    output: Path,
    *,
    full_cloud_cost_file: Path | None = None,
    require_full_cloud_cost: bool = False,
) -> int:
    tasks = discover_tasks(dataset_dir.resolve())
    cost_metadata = (
        _load_cost_metadata_from_baseline(full_cloud_cost_file)
        if full_cloud_cost_file is not None
        else _load_cost_metadata_from_manifest(output)
    )
    task_ids = {task.task_id for task in tasks}
    missing_costs = sorted(task_ids - set(cost_metadata))
    extra_costs = sorted(set(cost_metadata) - task_ids)
    if full_cloud_cost_file is not None and extra_costs:
        raise ValueError(f"full-cloud cost baseline contains unknown task ids: {extra_costs[:10]}")
    if require_full_cloud_cost and missing_costs:
        raise ValueError(f"training manifest is missing full-cloud costs for {len(missing_costs)} tasks: {missing_costs[:10]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fp:
        for task in tasks:
            metadata = {"task_dir": str(task.task_dir)}
            if task.task_id in cost_metadata:
                metadata.update(cost_metadata[task.task_id])
            row = {
                "prompt": [{"role": "user", "content": task.user_query}],
                "label": task.task_id,
                "metadata": metadata,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slime JSONL manifest from a ClawGym task directory")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-cloud-cost-file",
        type=Path,
        help="One-time baseline import; embeds each task's cost and provenance into metadata.",
    )
    parser.add_argument(
        "--require-full-cloud-cost",
        action="store_true",
        help="Fail if any discovered task lacks embedded full-cloud cost metadata.",
    )
    args = parser.parse_args()
    count = build_manifest(
        args.dataset_dir,
        args.output,
        full_cloud_cost_file=args.full_cloud_cost_file,
        require_full_cloud_cost=args.require_full_cloud_cost,
    )
    print(f"wrote {count} tasks to {args.output}")


if __name__ == "__main__":
    main()
