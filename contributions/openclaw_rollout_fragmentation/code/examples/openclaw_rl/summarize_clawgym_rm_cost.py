"""Summarize direct GLM-5.2 OpenClaw trajectories into a cost baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.openclaw_rl.prepare_clawgym_rm_e2b import payload_hash


@dataclass(frozen=True)
class Pricing:
    noncached_input: float = 0.0014
    cached_input: float = 0.00026
    output: float = 0.0044

    def as_dict(self) -> dict[str, float]:
        return {
            "noncached_input": self.noncached_input,
            "cached_input": self.cached_input,
            "output": self.output,
        }


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _first(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _integer(mapping[key])
    return 0


def trajectory_usage(path: Path, pricing: Pricing) -> dict[str, Any]:
    prompt_tokens = 0
    noncached_input_tokens = 0
    completion_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    assistant_turns = 0
    malformed_lines = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            message = event.get("message") if isinstance(event, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            completion = _first(usage, "output", "completion_tokens", "completionTokens")
            cache_read = _first(usage, "cacheRead", "cache_read_tokens", "cached_tokens", "cacheReadTokens")
            cache_write = _first(usage, "cacheWrite", "cache_write_tokens", "cacheWriteTokens")
            if "input" in usage:
                # pi-ai/OpenClaw normalizes `input` to uncached input. Its
                # totalTokens is input + cacheRead + cacheWrite + output.
                noncached = _integer(usage.get("input"))
                prompt = noncached + cache_read + cache_write
            else:
                # Raw OpenAI-compatible usage reports prompt_tokens inclusive
                # of cached tokens.
                prompt = _first(usage, "prompt_tokens", "promptTokens")
                cache_read = min(prompt, cache_read)
                cache_write = min(max(0, prompt - cache_read), cache_write)
                noncached = prompt - cache_read - cache_write
            prompt_tokens += prompt
            noncached_input_tokens += noncached
            completion_tokens += completion
            cache_read_tokens += cache_read
            cache_write_tokens += cache_write
            assistant_turns += 1

    noncached_cost = noncached_input_tokens * pricing.noncached_input / 1000.0
    cached_cost = cache_read_tokens * pricing.cached_input / 1000.0
    output_cost = completion_tokens * pricing.output / 1000.0
    return {
        "assistant_turns": assistant_turns,
        "prompt_tokens": prompt_tokens,
        "noncached_input_tokens": noncached_input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "noncached_input_cost_rmb": noncached_cost,
        "cached_input_cost_rmb": cached_cost,
        "output_cost_rmb": output_cost,
        "cost_rmb": noncached_cost + cached_cost + output_cost,
        "malformed_trajectory_lines": malformed_lines,
    }


def _load_tasks(dataset: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with dataset.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            task_id = str(row.get("task_id") or (row.get("metadata") or {}).get("task_id") or "")
            if not task_id:
                raise ValueError(f"dataset line {line_number} has no task_id")
            if task_id in seen_ids:
                raise ValueError(f"duplicate task_id in dataset: {task_id}")
            seen_ids.add(task_id)
            tasks.append({"task_id": task_id, "payload_hash": payload_hash(row)})
    if not tasks:
        raise ValueError(f"dataset is empty: {dataset}")
    return tasks


def _latest_attempts(output_dir: Path) -> dict[str, Path]:
    attempts: dict[str, tuple[float, Path]] = {}
    for metadata_path in output_dir.rglob("metadata.json"):
        trajectory = metadata_path.with_name("trajectory.jsonl")
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        digest = str(metadata.get("payload_hash") or "")
        if not digest:
            continue
        stamp = metadata_path.stat().st_mtime
        if digest not in attempts or stamp > attempts[digest][0]:
            attempts[digest] = (stamp, trajectory)
    return {digest: value[1] for digest, value in attempts.items()}


def _reward_score(trajectory: Path) -> float | None:
    reward_path = trajectory.with_name("reward.json")
    try:
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_score = reward.get("final_score")
    if raw_score is None and isinstance(reward.get("code_reward"), dict):
        raw_score = reward["code_reward"].get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    score = float(raw_score)
    return score if math.isfinite(score) else None


def summarize(
    dataset: Path,
    output_dir: Path,
    report_dir: Path,
    *,
    cloud_model: str,
    advisor_style: str,
    pricing: Pricing,
) -> dict[str, Any]:
    tasks = _load_tasks(dataset)
    attempts = _latest_attempts(output_dir)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        trajectory = attempts.get(task["payload_hash"])
        if trajectory is None:
            rows.append({**task, "status": "missing", "trajectory": ""})
            continue
        usage = trajectory_usage(trajectory, pricing)
        reward_score = _reward_score(trajectory)
        valid = usage["assistant_turns"] > 0 and usage["cost_rmb"] > 0 and math.isfinite(usage["cost_rmb"])
        rows.append(
            {
                **task,
                "status": "complete" if valid else "usage_missing",
                "trajectory": str(trajectory),
                "reward_score": reward_score if reward_score is not None else "",
                "reward_exact_pass": reward_score == 1.0 if reward_score is not None else "",
                **usage,
            }
        )

    complete = [row for row in rows if row["status"] == "complete"]
    missing = [row["task_id"] for row in rows if row["status"] != "complete"]
    scored = [row for row in complete if isinstance(row.get("reward_score"), float)]
    exact_passes = [row for row in scored if row["reward_exact_pass"]]
    token_fields = [
        "prompt_tokens",
        "noncached_input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    cost_fields = ["noncached_input_cost_rmb", "cached_input_cost_rmb", "output_cost_rmb", "cost_rmb"]
    summary = {
        "version": "openclaw-glm52-cost-report-v1",
        "cloud_model": cloud_model,
        "pricing_rmb_per_1k": pricing.as_dict(),
        "num_tasks": len(rows),
        "num_complete": len(complete),
        "num_incomplete": len(missing),
        "coverage": len(complete) / len(rows),
        "missing_task_ids": missing,
        "reward": {
            "num_scored": len(scored),
            "num_exact_pass": len(exact_passes),
            "exact_pass_rate": len(exact_passes) / len(scored) if scored else None,
            "mean_score": sum(row["reward_score"] for row in scored) / len(scored) if scored else None,
        },
        "totals": {
            **{field: sum(int(row.get(field, 0)) for row in complete) for field in token_fields},
            **{field: sum(float(row.get(field, 0.0)) for row in complete) for field in cost_fields},
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "task_id",
        "status",
        "reward_score",
        "reward_exact_pass",
        "assistant_turns",
        *token_fields,
        *cost_fields,
        "trajectory",
    ]
    with (report_dir / "task_costs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    baseline_path = report_dir / "full_cloud_costs.json"
    if not missing:
        baseline = {
            "version": "openclaw-full-cloud-cost-v1",
            "cloud_model": cloud_model,
            "advisor_style": advisor_style,
            "pricing_rmb_per_1k": pricing.as_dict(),
            "task_cost_rmb": {row["task_id"]: row["cost_rmb"] for row in complete},
            "samples_per_task": {row["task_id"]: 1 for row in complete},
            "mean_calls_per_task": {row["task_id"]: row["assistant_turns"] for row in complete},
            "num_tasks": len(complete),
            "num_rollouts": len(complete),
            "source": "direct GLM-5.2 OpenClaw E2B calibration",
        }
        temporary = baseline_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(baseline_path)
    elif baseline_path.exists():
        baseline_path.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="The complete, unfiltered RM E2B manifest.")
    parser.add_argument("--rollout-output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cloud-model", default="glm-5.2")
    parser.add_argument("--advisor-style", default="ace")
    parser.add_argument("--noncached-input-per-1k-rmb", type=float, default=0.0014)
    parser.add_argument("--cached-input-per-1k-rmb", type=float, default=0.00026)
    parser.add_argument("--output-per-1k-rmb", type=float, default=0.0044)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    pricing = Pricing(
        noncached_input=args.noncached_input_per_1k_rmb,
        cached_input=args.cached_input_per_1k_rmb,
        output=args.output_per_1k_rmb,
    )
    if any(not math.isfinite(value) or value < 0 for value in pricing.as_dict().values()):
        raise ValueError("all prices must be finite and non-negative")
    result = summarize(
        args.dataset.resolve(),
        args.rollout_output.resolve(),
        args.report_dir.resolve(),
        cloud_model=args.cloud_model,
        advisor_style=args.advisor_style,
        pricing=pricing,
    )
    console_result = {key: value for key, value in result.items() if key != "missing_task_ids"}
    print(json.dumps(console_result, ensure_ascii=False, sort_keys=True))
    if result["num_incomplete"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
