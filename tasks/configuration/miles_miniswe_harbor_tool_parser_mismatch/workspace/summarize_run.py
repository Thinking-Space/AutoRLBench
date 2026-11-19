#!/usr/bin/env python3
"""Aggregate Miles TITO-v2 leaf samples without assuming a fan-out cause."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = Path("/tmp/autorlbench-results")


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def load_samples(dump_dir: Path) -> list[dict[str, Any]]:
    import torch

    samples: list[dict[str, Any]] = []
    paths = sorted(dump_dir.glob("rollout_data/*.pt"))
    for path in paths:
        payload = torch.load(path, weights_only=False, map_location="cpu")
        raw_samples = payload.get("samples", [])
        if not isinstance(raw_samples, list):
            raise AssertionError(f"{path} does not contain a sample list")
        for sample in raw_samples:
            if not isinstance(sample, dict):
                raise AssertionError(f"{path} contains a non-dict sample")
            sample = dict(sample)
            sample["dump_file"] = str(path)
            samples.append(sample)
    return samples


def build_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaves: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        index = sample.get("index")
        if index is None:
            raise AssertionError("a rollout sample has no sample index")
        leaves[int(index)].append(sample)

    records: list[dict[str, Any]] = []
    for sample_index in sorted(leaves):
        leaf_rows = leaves[sample_index]
        metadata_rows = [row.get("metadata") or {} for row in leaf_rows]
        rewards = [value for row in leaf_rows if (value := _as_float(row.get("reward"))) is not None]
        response_lengths = [
            value
            for row in leaf_rows
            if (value := _as_float(row.get("response_length"))) is not None
        ]
        exit_statuses = [
            str(metadata.get("exit_status"))
            for metadata in metadata_rows
            if metadata.get("exit_status") is not None
        ]
        turns = [
            value
            for metadata in metadata_rows
            if (value := _as_float((metadata.get("agent_metrics") or {}).get("turns"))) is not None
        ]
        total_times = [
            value
            for metadata in metadata_rows
            if (value := _as_float((metadata.get("agent_metrics") or {}).get("total_time"))) is not None
        ]
        mismatches = [
            metadata.get("tito_session_mismatch")
            for metadata in metadata_rows
            if metadata.get("tito_session_mismatch") is not None
        ]
        trial_dirs = list(
            dict.fromkeys(
                metadata["trial_dir"]
                for metadata in metadata_rows
                if metadata.get("trial_dir")
            )
        )

        records.append(
            {
                "sample_index": sample_index,
                "group_index": leaf_rows[0].get("group_index"),
                "rollout_id": leaf_rows[0].get("rollout_id"),
                "leaf_count": len(leaf_rows),
                "statuses": [row.get("status") for row in leaf_rows],
                "exit_statuses": exit_statuses,
                "reward": _rounded(_mean(rewards)),
                "turns": _rounded(_mean(turns)),
                "response_length": _rounded(_mean(response_lengths)),
                "total_time": _rounded(_mean(total_times)),
                "tito_session_mismatch": mismatches,
                "has_tito_session_mismatch": bool(any(mismatches)),
                "trial_dirs": trial_dirs,
                "dump_files": list(
                    dict.fromkeys(
                        row["dump_file"] for row in leaf_rows if row.get("dump_file")
                    )
                ),
            }
        )
    return records


def build_summary(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    leaf_counts = [record["leaf_count"] for record in records]
    total_samples = len(samples)
    logical_rollouts = len(records)
    mismatch_records = [record for record in records if record["has_tito_session_mismatch"]]
    rewards = [record["reward"] for record in records if record["reward"] is not None]
    turns = [record["turns"] for record in records if record["turns"] is not None]
    total_times = [record["total_time"] for record in records if record["total_time"] is not None]
    exit_statuses = Counter(
        status
        for record in records
        for status in record["exit_statuses"]
    )
    statuses = Counter(
        status
        for sample in samples
        if (status := sample.get("status")) is not None
    )
    trial_dirs = {
        trial_dir
        for record in records
        for trial_dir in record["trial_dirs"]
    }

    expected_rollouts = int(os.environ.get("EXPECTED_LOGICAL_ROLLOUTS", "64"))
    return {
        "framework": "miles",
        "session_server": "v2",
        "message_matcher": "strict",
        "expected_logical_rollouts": expected_rollouts,
        "logical_rollouts": logical_rollouts,
        "training_samples": total_samples,
        "samples_per_logical_rollout": _rounded(total_samples / logical_rollouts)
        if logical_rollouts
        else None,
        "extra_training_samples": total_samples - logical_rollouts,
        "sample_inflation_ratio": _rounded(total_samples / logical_rollouts)
        if logical_rollouts
        else None,
        "fanout_rollouts": sum(count > 1 for count in leaf_counts),
        "max_leaves_per_logical_rollout": max(leaf_counts, default=0),
        "mismatch_rollouts": len(mismatch_records),
        "tito_session_mismatch_rate": _rounded(len(mismatch_records) / logical_rollouts)
        if logical_rollouts
        else None,
        "reward_mean": _rounded(_mean(rewards)),
        "positive_reward_rollouts": sum(value > 0 for value in rewards),
        "turns_mean": _rounded(_mean(turns)),
        "total_time_mean_seconds": _rounded(_mean(total_times)),
        "sample_statuses": dict(sorted(statuses.items())),
        "exit_statuses": dict(sorted(exit_statuses.items())),
        "harbor_trials": len(trial_dirs),
        "logical_rollout_count_complete": logical_rollouts == expected_rollouts,
    }


def main() -> int:
    results_dir = Path(
        os.environ.get("AUTORLBENCH_RESULTS", str(DEFAULT_RESULTS))
    )
    dump_dir = Path(
        os.environ.get("MILES_DUMP_DIR", str(results_dir / "dump_details"))
    )
    log_dir = results_dir / "experiment_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(dump_dir)
    if not samples:
        raise AssertionError(f"no Miles rollout samples found under {dump_dir}")

    records = build_records(samples)
    summary = build_summary(records, samples)
    summary["observation"] = (
        "fanout_observed" if summary["fanout_rollouts"] else "no_fanout_observed"
    )
    summary["records_path"] = str(log_dir / "rollout_records.jsonl")
    summary["dump_dir"] = str(dump_dir)

    records_path = log_dir / "rollout_records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    metrics_path = log_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["training_samples"] < summary["logical_rollouts"]:
        raise AssertionError("training sample count cannot be lower than logical rollout count")
    if not math.isfinite(summary["sample_inflation_ratio"] or 0):
        raise AssertionError("invalid sample inflation ratio")

    print(
        "PASS: observation complete; "
        f"training_samples/logical_rollouts={summary['sample_inflation_ratio']}x, "
        f"fanout_rollouts={summary['fanout_rollouts']}/{summary['logical_rollouts']}, "
        f"mismatch_rollouts={summary['mismatch_rollouts']}/{summary['logical_rollouts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
