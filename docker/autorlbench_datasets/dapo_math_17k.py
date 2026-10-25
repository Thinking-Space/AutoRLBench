"""BytedTsinghua-SIA/DAPO-Math-17k -> DAPO-Math-17k_verl/{train,test}.parquet

Shared by task_monkey_patch_011 / task_agent_loop_022 / task_hparam_overlong_023
(their launch scripts all want /workspace/DAPO-Math-17k_verl/{train,test}.parquet).

## Why we need our own split rule

Upstream is **a single split with no train/test**. The parquet on the Hub has 1791700 rows,
but it is the same 17917 problems concatenated 100 times (the authors confirmed on the
discussion board that this is an accidental duplication, not 1.79M problems).
The historical output was 16125 / 1792 (= 17917, ratio 10.0017%), but the seed and boundary
of that split are **recorded nowhere in the task**, not even in `agent_trace.json`. It cannot be
reproduced, so we first drop the 100 copies by row id, then apply a new, written-down split rule.

## Rule (v1)

    key    = extra_info.original_index or extra_info.index
               The actual upstream field is index, a UUID, unique per row and independent of row order.
               original_index is an earlier guess from the historical output; it still wins if present.
    digest = sha256(b"autorlbench-dapo-split-v1:" + key)
    test   when int(digest[:16], 16) % 10 == 0, otherwise train

A hash instead of `random.shuffle(seed=42)` because a hash split is **independent of row order
and library versions**. `datasets.train_test_split` and `numpy.random.permutation` both drift
across versions, and "same upstream + same script = same bytes" is the whole point of this asset.
The `v1` in the salt is reserved for future rule changes: bump to v2, never change semantics in place.

Expect ~10% to land in test. **This will not equal the historical 16125/1792**, so
historical_final_baseline / required_final_score in the 011/022/023 grader_spec.json files must be
recomputed. (A known and approved consequence, not a bug.)

## Upstream schema

Upstream should already be in verl format (data_source='math_dapo', style='rule-lighteval/MATH_v2'),
based on the schema actually dumped from the historical output. **But this machine has no network,
so upstream was not checked first-hand.** Every required field is therefore checked explicitly
below, and a mismatch stops with an error instead of guessing onward.

Usage:
    python3 -m autorlbench_datasets.dapo_math_17k --out-dir ./prepared/DAPO-Math-17k_verl
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ._common import hf_datasets_cache, report, write_split

REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
# Pin the revision. This repo's main has been changed before and may be again; the row-count
# assertion catches "the count changed" but not "all 17917 ids remain, problem text swapped".
REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DATA_SOURCE = "math_dapo"  # reward_score/__init__.py:142 matches this string exactly

SPLIT_SALT = b"autorlbench-dapo-split-v1:"
TEST_MODULUS = 10  # 1/10 goes to test

# Row count after dedup, not the parquet's physical row count (which is this times 100).
EXPECTED_TOTAL = 17917  # historical 16125 + 1792; a mismatch means upstream changed

# Part of the cache key (see docker/autorlbench_dataset.py). **Bump this whenever the recipe
# changes**: if the split rule, schema checks, or write path change, the previously cached
# parquet is no longer this recipe's output. Without a bump, whichever of the three tasks runs
# first keeps getting stale bytes, and "same recipe, same data" is why these files exist.
# v2: drop the 100x duplication by extra_info.index first, then split on that UUID. v1 compared
# all 1791700 raw rows against 17917 and fell back to problem text when original_index was missing.
RECIPE_VERSION = "2"


def split_of(key: str) -> str:
    digest = hashlib.sha256(SPLIT_SALT + key.encode("utf-8")).hexdigest()
    return "test" if int(digest[:16], 16) % TEST_MODULUS == 0 else "train"


def row_id(row: dict) -> str | None:
    """The UUID upstream uses to identify a problem.

    The field in the file is extra_info.index. original_index is a name guessed from the
    historical output and does not exist at this revision; both are accepted so that an
    upstream rename does not silently fall back to problem text.
    """
    extra = row.get("extra_info") or {}
    key = extra.get("original_index") or extra.get("index")
    return str(key) if key else None


def row_fingerprint(row: dict) -> bytes:
    """Copies of one id must match field by field. Covers only the fields the split and scorer read."""
    prompt = row.get("prompt") or [{}]
    content = ""
    if prompt and isinstance(prompt[0], dict):
        content = prompt[0].get("content") or ""
    reward = row.get("reward_model") or {}
    digest = hashlib.sha256()
    for part in (
        row.get("data_source") or "",
        row.get("ability") or "",
        reward.get("ground_truth") or "",
        reward.get("style") or "",
        content,
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def dedupe_copies(rows_iter) -> tuple[list[dict], int]:
    """Drop the 100x duplication, keeping the first row seen for each id.

    Order is that of the first copy. If copies of one id differ, stop: keeping the first
    would swallow "upstream changed some of the copies".
    """
    kept: list[dict] = []
    seen: dict[str, bytes] = {}
    total = 0
    for row in rows_iter:
        total += 1
        key = row_id(row)
        if not key:
            raise SystemExit(
                "upstream row has no extra_info.index / original_index; cannot drop the 100x duplication."
            )
        fingerprint = row_fingerprint(row)
        previous = seen.get(key)
        if previous is None:
            seen[key] = fingerprint
            kept.append(row)
        elif previous != fingerprint:
            raise SystemExit(
                f"copies of extra_info.index={key} differ; cannot just keep the first."
            )
    return kept, total


def split_key(row: dict, fallback_used: list[bool]) -> str:
    """Pick a stable row identifier.

    Prefers the upstream UUID. If upstream lacks that field, fall back to the problem text:
    also independent of row order, but duplicate problem texts land on the same side. The
    fallback is reported at the end, not silently.
    """
    key = row_id(row)
    if key:
        return key
    fallback_used[0] = True
    prompt = row["prompt"]
    return prompt[0]["content"]


def check_schema(row: dict) -> None:
    if row.get("data_source") != DATA_SOURCE:
        raise SystemExit(
            f"upstream data_source is {row.get('data_source')!r}, expected {DATA_SOURCE!r}.\n"
            f"The scorer routes on this exact string (reward_score/__init__.py:142); it cannot change."
        )
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or "content" not in prompt[0]:
        raise SystemExit(
            f"upstream prompt is not a list of chat messages: {type(prompt).__name__}.\n"
            f"RLHFDataset requires example['prompt'] to be a list (rl_dataset.py:313)."
        )
    rm = row.get("reward_model") or {}
    if "ground_truth" not in rm:
        raise SystemExit(
            "upstream reward_model has no ground_truth"
            " (reward_manager/dapo.py:92 reads this key directly)."
        )


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """Download upstream, split by the rule, write both parquets; return the written paths.

    Kept separate from main() because the in-container `autorlbench_dataset` calls it inside
    the **cache directory** and then hardlinks the output to where the task expects it; it
    needs a return value, not a process exit code.
    """
    # Before the import: see _common.hf_cache_root(). Without cache_dir, upstream lands in
    # ~/.cache/huggingface, which is the container's writable layer: nothing shows up in the
    # host's datasets/, and the three duplicate downloads 011 / 022 / 023 meant to avoid come back.
    cache_dir = hf_datasets_cache()
    from datasets import load_dataset

    print(f"downloading {REPO}@{REVISION[:12]} ...")
    dataset = load_dataset(REPO, cache_dir=str(cache_dir), revision=REVISION)
    if len(dataset.keys()) != 1:
        print(f"!! upstream has multiple splits: {list(dataset.keys())}; merging all and re-splitting")
    raw = (dict(record) for split in dataset for record in dataset[split])
    rows, raw_count = dedupe_copies(raw)

    if not rows:
        raise SystemExit("upstream has no rows")
    check_schema(rows[0])
    if raw_count != len(rows):
        print(f"  dropped duplicate copies: {raw_count} rows -> {len(rows)} rows")

    if len(rows) != EXPECTED_TOTAL:
        msg = (
            f"{len(rows)} rows after dedup ({raw_count} raw), "
            f"historical output totals {EXPECTED_TOTAL}"
        )
        if not allow_count_mismatch:
            raise SystemExit(f"{msg}. Upstream changed; once confirmed, add --allow-count-mismatch.")
        print(f"!! {msg} (--allow-count-mismatch is set, continuing)")

    fallback_used = [False]
    buckets: dict[str, list[dict]] = {"train": [], "test": []}
    for row in rows:
        buckets[split_of(split_key(row, fallback_used))].append(row)

    written = []
    for split_name in ("train", "test"):
        out = []
        for idx, row in enumerate(buckets[split_name]):
            extra = dict(row.get("extra_info") or {})
            extra["index"] = idx
            extra["split"] = split_name
            out.append({**row, "extra_info": extra})
        path = write_split(out, out_dir, split_name)
        pct = 100.0 * len(out) / len(rows)
        print(f"  {split_name}: {len(out)} rows ({pct:.2f}%) -> {path}")
        written.append(path)

    if fallback_used[0]:
        print(
            "\n!! upstream has no extra_info.index / original_index; the split fell back to problem text.\n"
            "   The rule is still deterministic, but duplicate problem texts land on the same side."
            " Please note this."
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    written = build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)

    report(written)
    print(
        "\nReminder: this split does not equal the historical 16125/1792; recompute\n"
        "historical_final_baseline / required_final_score in the 011/022/023 grader_spec.json files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
