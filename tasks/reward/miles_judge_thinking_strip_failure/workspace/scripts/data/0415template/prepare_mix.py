"""
Build mix4k_union jsonl by merging two 0415template SFT subsets.

Source (read-only)::

    /user/lizhen/data/autocuda_sft/0415template/raw/sft_data_megatron.sample_4000.jsonl
    /user/lizhen/data/autocuda_sft/0415template/raw/sft_data_repeat5033.sample_4000.jsonl

Output (writable)::

    /user/luozhenghang/0226_miles/data/sft/0415template/raw/sft_data_mix4k_union.sample_6264.jsonl

Why this exists:
- megatron 4k and repeat5033 4k share ~1736 questions; for those overlapping
  questions both sides emit byte-identical responses (cross-checked here, expect 0
  divergent).
- Union by question SHA1 yields 4000 + (4000 - 1736) = 6264 unique rows.
- Megatron-first ordering keeps the file deterministic across re-runs.

Run::

    python3 scripts/data/0415template/prepare_mix.py

Optional flags via env::

    MIX_OUT=/abs/path/to/output.jsonl python3 ...
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

SRC_MEGA = Path("/user/lizhen/data/autocuda_sft/0415template/raw/sft_data_megatron.sample_4000.jsonl")
SRC_REPEAT = Path("/user/lizhen/data/autocuda_sft/0415template/raw/sft_data_repeat5033.sample_4000.jsonl")
DEFAULT_OUT = Path(
    "/user/luozhenghang/0226_miles/data/sft/0415template/raw/sft_data_mix4k_union.sample_6264.jsonl"
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _qhash(rec: dict) -> str:
    return hashlib.sha1(rec.get("question", "").encode("utf-8")).hexdigest()


def main() -> int:
    out_path = Path(os.environ.get("MIX_OUT", str(DEFAULT_OUT)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not SRC_MEGA.exists() or not SRC_REPEAT.exists():
        print(f"missing source: {SRC_MEGA} or {SRC_REPEAT}", file=sys.stderr)
        return 1

    t0 = time.time()
    mega = _load_jsonl(SRC_MEGA)
    repeat = _load_jsonl(SRC_REPEAT)
    print(f"loaded mega={len(mega)}  repeat={len(repeat)}  in {time.time() - t0:.1f}s")

    seen: set[str] = set()
    merged: list[dict] = []
    mega_idx = {_qhash(r): r for r in mega}

    for r in mega:
        h = _qhash(r)
        if h in seen:
            continue
        seen.add(h)
        merged.append(r)
    n_only_mega = len(merged)

    overlap_with_diff_response = 0
    added_from_repeat = 0
    for r in repeat:
        h = _qhash(r)
        if h in seen:
            other = mega_idx.get(h)
            if other is not None and other.get("response") != r.get("response"):
                overlap_with_diff_response += 1
            continue
        seen.add(h)
        merged.append(r)
        added_from_repeat += 1

    print(f"unique total       : {len(merged)}")
    print(f"  from megatron     : {n_only_mega}")
    print(f"  added from repeat : {added_from_repeat}")
    print(f"  overlap (skipped) : {len(repeat) - added_from_repeat}")
    print(f"  overlap-but-resp-diff: {overlap_with_diff_response}  (sanity, expect 0)")

    if overlap_with_diff_response != 0:
        print("WARN: overlap responses diverged; verify before training.", file=sys.stderr)

    with open(out_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nwrote: {out_path}  ({size_mb:.1f} MB)  in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
