#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.group_adapter import (  # noqa: E402
    accept_valid_group,
    mark_generator_zero,
    normalize_group,
)
from runtime.model import Sample  # noqa: E402


def build_group() -> list[Sample]:
    samples = [
        Sample(
            index=index,
            group_index=17,
            prompt="Complete the assigned target.",
            label={"target": "example"},
            metadata={"group_valid": True},
        )
        for index in range(8)
    ]
    for sample in samples[3:]:
        mark_generator_zero(sample)
    return samples


def main() -> None:
    samples = build_group()
    values = normalize_group(
        samples,
        lambda eligible: [1.0 if index % 2 == 0 else 0.0
                          for index, _ in enumerate(eligible)],
    )
    keep, reason = accept_valid_group(samples)
    summary = {
        "group_size": len(samples),
        "valid_samples": 8,
        "behavioral_zeros": 5,
        "submitted_answers": 3,
        "rewards": values,
        "binary_records": sum(
            sample.metadata.get("reward_record", {}).get("mode") == "binary"
            for sample in samples
        ),
        "keep": keep,
        "drop_reason": reason,
    }
    print(json.dumps(summary, indent=2))
    if keep:
        raise SystemExit(0)
    raise SystemExit(
        "reproduced: valid mixed group was dropped because five "
        "generator-assigned zeros bypassed reward normalization"
    )


if __name__ == "__main__":
    main()
