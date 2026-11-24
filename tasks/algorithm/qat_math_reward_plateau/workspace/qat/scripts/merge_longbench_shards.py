#!/usr/bin/env python3
"""Merge disjoint LongBench v2 shard reports into one evaluator report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CONTRACT_FIELDS = (
    "evaluator_schema_version",
    "evaluator_source_sha256",
    "benchmark",
    "dataset_id",
    "dataset_split",
    "subset",
    "difficulty",
    "profile",
    "instruction",
    "expected_rows",
    "source_count",
    "model_path_real",
    "data",
    "kv_bits",
    "dtype_resolved",
    "attn_implementation",
    "preserve_projection_bias",
    "exact_bias_free_recompute",
    "seed",
    "thinking_enabled",
    "prompt_token_limit",
    "sampling",
    "num_shards",
)


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one shard report is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = reports[0]
    expected_shards = int(first["num_shards"])
    if len(reports) != expected_shards:
        raise ValueError(f"found {len(reports)} reports, expected {expected_shards}")

    for report in reports[1:]:
        for field in CONTRACT_FIELDS:
            if report.get(field) != first.get(field):
                raise ValueError(f"shard reports disagree on {field}")

    shard_indices = [int(report["shard_index"]) for report in reports]
    if sorted(shard_indices) != list(range(expected_shards)):
        raise ValueError("shard indices are incomplete or duplicated")

    predictions = [prediction for report in reports for prediction in report["predictions"]]
    predictions.sort(key=lambda prediction: int(prediction["index"]))
    expected_rows = int(first["expected_rows"])
    if [int(prediction["index"]) for prediction in predictions] != list(range(expected_rows)):
        raise ValueError("merged predictions do not cover every benchmark row exactly once")

    count = len(predictions)
    correct = sum(int(prediction["correct"]) for prediction in predictions)
    truncated = sum(int(prediction["truncated"]) for prediction in predictions)
    eos_samples = sum(int(prediction["eos_seen"]) for prediction in predictions)
    extraction_failures = sum(prediction.get("prediction") is None for prediction in predictions)
    prompt_tokens = [int(prediction["prompt_tokens"]) for prediction in predictions]
    response_tokens = [int(prediction["response_tokens"]) for prediction in predictions]

    merged = dict(first)
    merged.update(
        {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "shard_index": None,
            "merged_shards": [str(path.resolve()) for path in paths],
            "count": count,
            "correct": correct,
            "accuracy": correct / count,
            "total_samples": count,
            "truncated_samples": truncated,
            "truncated_fraction": truncated / count,
            "eos_samples": eos_samples,
            "eos_fraction": eos_samples / count,
            "extraction_failures": extraction_failures,
            "extraction_failure_fraction": extraction_failures / count,
            "mean_prompt_tokens": sum(prompt_tokens) / count,
            "max_prompt_tokens": max(prompt_tokens),
            "mean_response_tokens": sum(response_tokens) / count,
            "max_response_tokens": max(response_tokens),
            "elapsed_seconds": max(float(report["elapsed_seconds"]) for report in reports),
            "predictions": predictions,
        }
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge_reports(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": report["count"], "accuracy": report["accuracy"]}))


if __name__ == "__main__":
    main()
