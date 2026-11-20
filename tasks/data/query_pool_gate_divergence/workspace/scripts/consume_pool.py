"""Stand-in for the trainer: read the pool exactly as a run would, and record what it got.

The real run streams prompts out of `TRAIN_FILE` in shuffled order, takes
`OVER_SAMPLING_BATCH_SIZE` prompts per rollout, keeps `ROLLOUT_BATCH_SIZE` of them after the
dynamic-sampling filter, and samples `N_SAMPLES_PER_PROMPT` trajectories each. Nothing here
needs a GPU to show which prompts a rollout would have trained on, and that is what the gate's
decisions are ultimately about.

Writes `$RUN_LOG_DIR/run.json` (default `$AUTORLBENCH_RESULTS/experiment_logs/run.json`).
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys
from pathlib import Path


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


def main() -> int:
    train_file = Path(os.environ["TRAIN_FILE"])
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    identities = [
        (row.get("metadata", {}).get("pool_source"), row.get("metadata", {}).get("instance_id"))
        for row in rows
    ]

    rollouts = env_int("NUM_ROLLOUT", 4)
    over_sampling = env_int("OVER_SAMPLING_BATCH_SIZE", 64)
    batch_size = env_int("ROLLOUT_BATCH_SIZE", 16)
    samples = env_int("N_SAMPLES_PER_PROMPT", 8)
    seed = env_int("ROLLOUT_SEED", 42)

    order = list(range(len(identities)))
    if os.environ.get("ROLLOUT_SHUFFLE", "0") == "1":
        random.Random(seed).shuffle(order)

    cursor = 0
    batches = []
    for rollout_id in range(rollouts):
        picked = []
        while len(picked) < over_sampling:
            if cursor >= len(order):
                cursor = 0  # the stream wraps: a rollout wider than the pool repeats prompts
            picked.append(order[cursor])
            cursor += 1
        composition = collections.Counter(identities[index][0] for index in picked)
        unique = len({identities[index] for index in picked})
        batches.append({
            "rollout_id": rollout_id,
            "prompts": len(picked),
            "unique_prompts": unique,
            "duplicate_prompts": len(picked) - unique,
            "kept_after_filter": min(batch_size, len(picked)),
            "trajectories": min(batch_size, len(picked)) * samples,
            "source_mix": dict(sorted(composition.items())),
        })

    out_dir = Path(os.environ.get("RUN_LOG_DIR") or
                   (Path(os.environ.get("AUTORLBENCH_RESULTS", ".")) / "experiment_logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_tag": os.environ.get("RUN_TAG", ""),
        "project": os.environ.get("PROJECT_NAME", ""),
        "train_file": str(train_file),
        "pool_total_env": os.environ.get("POOL_TOTAL", ""),
        "rows_read": len(rows),
        "unique_rows": len(set(identities)),
        "pool_source_counts": dict(sorted(collections.Counter(source for source, _ in identities).items())),
        "rollout_config": {
            "num_rollout": rollouts,
            "over_sampling_batch_size": over_sampling,
            "rollout_batch_size": batch_size,
            "n_samples_per_prompt": samples,
            "shuffle_seed": seed,
        },
        "batches": batches,
        "duplicate_prompts_total": sum(batch["duplicate_prompts"] for batch in batches),
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"consumed {len(rows)} prompts into {len(batches)} rollouts as {record['run_tag']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
