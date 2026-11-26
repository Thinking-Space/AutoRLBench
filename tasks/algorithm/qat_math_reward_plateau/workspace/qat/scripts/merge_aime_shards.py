#!/usr/bin/env python3
"""Merge disjoint AIME evaluator shard reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_FIELDS = (
    "evaluator_schema_version", "evaluator_source_sha256", "model_path_real", "data",
    "kv_bits", "dtype_resolved", "instruction", "thinking_enabled", "sampling",
    "profile", "source_count", "num_shards",
)


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one shard report is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = reports[0]
    shard_count = int(first["num_shards"])
    if len(reports) != shard_count:
        raise ValueError(f"found {len(reports)} reports, expected {shard_count}")
    for report in reports[1:]:
        for field in CONTRACT_FIELDS:
            if report.get(field) != first.get(field):
                raise ValueError(f"shard reports disagree on {field}")
    if sorted(int(report["shard_index"]) for report in reports) != list(range(shard_count)):
        raise ValueError("shard indices are incomplete or duplicated")
    predictions = sorted(
        (prediction for report in reports for prediction in report["predictions"]),
        key=lambda prediction: int(prediction["index"]),
    )
    source_count = int(first["source_count"])
    if [int(prediction["index"]) for prediction in predictions] != list(range(source_count)):
        raise ValueError("merged predictions do not cover every AIME problem exactly once")
    generated = [sample for prediction in predictions for sample in prediction["samples"]]
    correct = sum(float(prediction["sample_accuracy"]) for prediction in predictions)
    truncated = sum(bool(sample["truncated"]) for sample in generated)
    eos = sum(bool(sample["eos_seen"]) for sample in generated)
    failures = sum(sample["prediction"] is None for sample in generated)
    report = dict(first)
    report.update(
        {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "shard_index": None,
            "merged_shards": [str(path.resolve()) for path in paths],
            "count": len(predictions),
            "correct": correct,
            "correct_samples_normalized": correct,
            "accuracy": correct / len(predictions),
            "total_samples": len(generated),
            "truncated_samples": truncated,
            "truncated_fraction": truncated / len(generated),
            "eos_samples": eos,
            "eos_fraction": eos / len(generated),
            "extraction_failures": failures,
            "extraction_failure_fraction": failures / len(generated),
            "mean_response_tokens": sum(int(sample["response_tokens"]) for sample in generated) / len(generated),
            "max_response_tokens": max((int(sample["response_tokens"]) for sample in generated), default=0),
            "predictions": predictions,
            "elapsed_seconds": max(float(item["elapsed_seconds"]) for item in reports),
        }
    )
    return report


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
