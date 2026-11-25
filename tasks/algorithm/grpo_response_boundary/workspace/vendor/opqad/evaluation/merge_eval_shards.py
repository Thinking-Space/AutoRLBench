#!/usr/bin/env python3
"""Merge sharded eval_vllm_turboquant.py outputs into one result.

Sharding splits by PROBLEM (`--shard i --num-shards k` keeps problems
i, i+k, i+2k, ...), so every shard holds the complete set of n samples for the
problems it owns.  Merging is therefore a concatenation plus a re-derivation of
the aggregates -- never an average of averages, which would be wrong the moment
the shards hold different numbers of problems (30 problems over 4 shards is
8/8/7/7).

What it checks before merging, because a silently wrong merge is worse than a
missing one:

  * every shard ran the same model, kv-cache dtype, context, n and seed
  * the shards' problem indices are disjoint and, together, contiguous from 0
  * no shard is missing (num_shards taken from the files, cross-checked)

Usage:
  merge_eval_shards.py --out merged.json shard0.json shard1.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Fields that describe the experiment.  If any of these differ across shards,
# the shards are not parts of one run and must not be added together.
IDENTITY = (
    "benchmark", "model", "kv_cache_dtype_requested", "max_model_len",
    "dtype", "tensor_parallel_size", "subset", "data",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge even if some problem indices are missing")
    a = ap.parse_args()

    loaded = []
    for p in a.shards:
        with open(p) as f:
            loaded.append((p, json.load(f)))

    base = loaded[0][1]
    for p, d in loaded[1:]:
        for k in IDENTITY:
            if d.get(k) != base.get(k):
                raise SystemExit(
                    f"{p}: {k}={d.get(k)!r} but {loaded[0][0]} has "
                    f"{base.get(k)!r}.  These are not shards of one run."
                )
        if d.get("sampling") != base.get("sampling"):
            raise SystemExit(
                f"{p}: sampling parameters differ from {loaded[0][0]}; "
                "merging would mix two different experiments."
            )

    preds = []
    seen: dict[int, Path] = {}
    for p, d in loaded:
        for rec in d["predictions"]:
            idx = rec["index"]
            if idx in seen:
                raise SystemExit(
                    f"problem index {idx} appears in both {seen[idx]} and {p}: "
                    "the shards overlap, so totals would double-count."
                )
            seen[idx] = p
            preds.append(rec)
    preds.sort(key=lambda r: r["index"])

    idxs = [r["index"] for r in preds]
    expected = list(range(len(idxs)))
    if idxs != expected and not a.allow_partial:
        missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
        raise SystemExit(
            f"problem indices are not contiguous from 0; missing {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}.  A shard did not finish.  "
            "Re-run it, or pass --allow-partial if a partial result is what you "
            "want -- but then the accuracy is over a subset, not the benchmark."
        )

    # Re-derive every aggregate from the samples.  None of these are averaged
    # from the shards' own aggregates.
    total = correct = trunc = eos = extraction_failures = 0
    ptok: list[int] = []
    rtok: list[int] = []
    for rec in preds:
        for s in rec["samples"]:
            total += 1
            correct += int(bool(s.get("correct")))
            trunc += int(bool(s.get("truncated")))
            eos += int(not s.get("truncated"))
            if s.get("prediction") in (None, ""):
                extraction_failures += 1
            if s.get("prompt_tokens") is not None:
                ptok.append(int(s["prompt_tokens"]))
            if s.get("response_tokens") is not None:
                rtok.append(int(s["response_tokens"]))

    out = dict(base)
    out["predictions"] = preds
    out["count"] = len(preds)
    out["total_samples"] = total
    out["correct"] = correct
    out["accuracy"] = correct / total if total else 0.0
    out["truncated_samples"] = trunc
    out["truncated_fraction"] = trunc / total if total else 0.0
    out["eos_samples"] = eos
    out["eos_fraction"] = eos / total if total else 0.0
    out["extraction_failures"] = extraction_failures
    if ptok:
        out["mean_prompt_tokens"] = sum(ptok) / len(ptok)
        out["max_prompt_tokens"] = max(ptok)
    if rtok:
        out["mean_response_tokens"] = sum(rtok) / len(rtok)
        out["max_response_tokens"] = max(rtok)
    # Wall clock of a sharded run is the slowest shard, not the sum: the shards
    # ran concurrently.  Keep both so neither reading is guessed at.
    el = [d.get("elapsed_seconds", 0.0) for _, d in loaded]
    out["elapsed_seconds"] = max(el)
    out["elapsed_seconds_sum"] = sum(el)
    out["merged_from"] = [str(p) for p, _ in loaded]
    out["merged_shard_count"] = len(loaded)
    # `out` was seeded from one shard's dict, so it still carries that shard's
    # own index.  A merged file that says `shard: 0` invites being read as
    # shard 0; drop it and leave `num_shards` alone, since that is what
    # `merged_shard_count` has to be checked against.
    out.pop("shard", None)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"merged {len(loaded)} shards -> {a.out}")
    print(f"  problems {len(preds)}  samples {total}")
    print(f"  accuracy {out['accuracy']:.4f}  eos {out['eos_fraction']:.3f}  "
          f"truncated {out['truncated_fraction']:.3f}")
    print(f"  mean response tokens {out.get('mean_response_tokens', 0):.0f}")
    print(f"  wall clock {out['elapsed_seconds']:.0f}s (slowest shard; "
          f"{out['elapsed_seconds_sum']:.0f}s of GPU time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
