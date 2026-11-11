#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.fully_async import clone_group_for_retry  # noqa: E402
from runtime.generate import run_generation  # noqa: E402
from runtime.sample import Sample  # noqa: E402


def make_sample() -> Sample:
    return Sample(
        index=7,
        group_index=17,
        prompt=[
            {"role": "system", "content": "Use tools carefully."},
            {"role": "user", "content": "Complete the assigned target."},
        ],
        label={"target": "example"},
    )


def main() -> None:
    original = run_generation(make_sample())
    first_error = None
    consecutive_failures = 0

    for group_id in range(128):
        retry = clone_group_for_retry([original])[0]
        try:
            run_generation(retry)
        except ValueError as exc:
            first_error = first_error or str(exc)
            consecutive_failures += 1
            if group_id < 3:
                print(
                    f"replayed group {group_id}: "
                    f"{consecutive_failures}/128 consecutive failures "
                    f"reason={exc}"
                )

    summary = {
        "completed_updates_before_failure": 160,
        "prompt_type_after_generation": type(original.prompt).__name__,
        "first_replay_error": first_error,
        "consecutive_failed_replays": consecutive_failures,
    }
    print(json.dumps(summary, indent=2))
    if consecutive_failures == 128:
        raise SystemExit(
            "reproduced: retry reset preserved rendered prompt; "
            "128 consecutive replays failed before generation"
        )


if __name__ == "__main__":
    main()
