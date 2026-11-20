#!/usr/bin/env python3
"""Emit one profile's settings as shell assignments.

The profile table in ``preflight_mopd.py`` is the single definition of what a
profile means.  Bash consumers used to each carry their own ``case "$PROFILE"``
block, so the model path, context window, concurrency, tensor-parallel width
and avg@N drifted apart between them -- a 1.5B run launched through the
watcher silently inherited the 7B's evaluation protocol on all four.

Usage:
    eval "$(python scripts/profile_env.py qwen_brorl_1p5b)"
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight_mopd import _profiles  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} PROFILE", file=sys.stderr)
        return 2
    profiles = _profiles()
    name = argv[1]
    if name not in profiles:
        print(f"unknown PROFILE={name}; known: {', '.join(sorted(profiles))}", file=sys.stderr)
        return 2
    p = profiles[name]
    for key, value in (
        ("PROFILE_MODEL", p.model),
        ("PROFILE_TEACHER", p.teacher),
        ("PROFILE_TRAIN_PARQUET", p.prepared_parquet),
        ("BENCHMARK", p.eval_benchmark),
        ("BASE_MODEL", p.model),
        ("DATA", p.eval_data),
        ("MAX_MODEL_LEN", p.eval_max_model_len),
        ("MAX_NUM_SEQS", p.eval_max_num_seqs),
        ("TENSOR_PARALLEL_SIZE", p.eval_tensor_parallel_size),
        ("N", p.eval_num_samples),
        ("PROMPT_LENGTH", p.prompt_length),
        ("RESPONSE_LENGTH", p.response_length),
    ):
        print(f"{key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
