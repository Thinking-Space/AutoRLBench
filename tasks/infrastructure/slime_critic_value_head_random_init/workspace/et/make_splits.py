#!/usr/bin/env python3
"""Split the admitted Endless Terminals prompt set into train / val / test.

A run must train on one set of tasks and be scored on another, and the split
must be stable across runs, or a comparison between runs measures the split
instead of the change under study. So the split is a pure function of the
admitted task list and a seed: run this at any time and you get the same three
files.

Sizing (defaults) comes from a power calculation for the end-to-end held-out
pass-rate claim:

  test  300 tasks x 4 samples   Scored *paired* -- the same tasks and the same
                                per-task sample budget each time, so task
                                difficulty cancels out of the difference.
                                With a within-task correlation of ~0.6 that is
                                ~430 effective tasks, enough to resolve a 5-6
                                percentage-point difference in pass rate at
                                80% power. Going past 4 samples per task buys
                                almost nothing: at rho=0.6, k=8 is only 8%
                                better than k=4 for twice the generation.
  val   150 tasks x 2 samples   Used to decide when training has converged, so
                                that the stopping point is not chosen on the
                                test set.
  train everything else         ~1k tasks; at 16 episodes per rollout the
                                trainer sees each one about once per 65
                                rollouts.

Usage:
    python3 -m et.make_splits --prompts /tmp/et_prompts/endless_terminals.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="endless_terminals.jsonl from et.build_dataset")
    ap.add_argument("--val-size", type=int, default=150)
    ap.add_argument("--test-size", type=int, default=300)
    # Fixed by default. A split that moved between runs would silently turn a
    # within-task comparison into a between-task one.
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    # Sort before shuffling: the input order is whatever the validation pass
    # happened to emit, and sorting makes the shuffle depend on the seed alone.
    rows.sort(key=lambda r: r["label"])

    if args.val_size + args.test_size >= len(rows):
        raise SystemExit(
            f"{len(rows)} admitted tasks cannot supply a {args.val_size}+{args.test_size} holdout"
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    test, val, train = (
        rows[: args.test_size],
        rows[args.test_size : args.test_size + args.val_size],
        rows[args.test_size + args.val_size :],
    )

    out_dir = os.path.dirname(os.path.abspath(args.prompts))
    stem = os.path.splitext(os.path.basename(args.prompts))[0]
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(out_dir, f"{stem}_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in sorted(part, key=lambda r: r["label"]):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:>5}: {len(part):5d} tasks -> {path}")

    manifest = os.path.join(out_dir, "splits.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": args.seed,
                "n_admitted": len(rows),
                "train": sorted(r["label"] for r in train),
                "val": sorted(r["label"] for r in val),
                "test": sorted(r["label"] for r in test),
            },
            fh,
            indent=2,
        )
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
