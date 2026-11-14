#!/usr/bin/env python3
"""Shared readers for the two public TACO sources this task is built on.

Sources (both public on HF, fetched verbatim by environment/prepare.sh):

* ``BAAI/TACO`` split ``ALL``, train shards ``train-0000X-of-00009.parquet``
  -- the native, unfiltered training pool (25,443 rows).
* ``likaixin/TACO-verified`` file ``taco_verified.json`` -- the execution-verified
  subset (12,898 rows): every kept problem has at least one bundled solution
  that passes all of its own tests within 10 s.

The verified file is a single 1.9 GB JSON array whose records embed every
solution, so it is streamed record-by-record instead of loaded whole. Its
records carry an ``id`` plus the full TACO row (same schema); the loader below
verifies the id <-> TACO train row mapping by question text before trusting it
(see :func:`iter_verified_records` / :func:`verify_id_mapping`).

This module is used both by the task lifecycle (environment/assemble_dataset.py,
run inside the prepare stage) and by the author-side purification pipeline
(environment/data_prep/). It is stdlib + pyarrow only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterator

TACO_TRAIN_SHARD_GLOB = "train-*-of-00009.parquet"
TACO_TEST_SHARD_GLOB = "test-*-of-*.parquet"
VERIFIED_FILE = "taco_verified.json"

# TACO's `solutions` column is a JSON string holding a list of code strings.
# A handful of rows store a plain list instead; normalize both.


def parse_solutions(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s) for s in raw]
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(val, list):
        return [str(s) for s in val]
    return []


def code_type_of(input_output: str | None) -> str:
    """Mirror taco.py's dispatch: fn_name present -> call_based, else stdin."""
    try:
        io = json.loads(input_output) if input_output else None
    except (TypeError, ValueError):
        return "unknown"
    if not isinstance(io, dict):
        return "unknown"
    return "call_based" if io.get("fn_name") else "stdin"


def n_tests_of(input_output: str | None) -> int:
    try:
        io = json.loads(input_output) if input_output else None
    except (TypeError, ValueError):
        return 0
    if not isinstance(io, dict):
        return 0
    return len(io.get("inputs") or [])


def iter_taco_train(sources_dir: str | Path, with_solutions: bool = True) -> Iterator[dict]:
    """Yield TACO ALL/train rows in a stable order, tagged with their index.

    The index is the position in the concatenated shard order (shards sorted by
    name). TACO-verified ids are 1-based positions in this same order -- the
    mapping is verified empirically by verify_id_mapping() before any selection
    trusts it.

    with_solutions=False skips the (large) solutions column: the assembler never
    needs it and it roughly doubles memory.
    """
    import pyarrow.parquet as pq

    root = Path(sources_dir)
    shards = sorted(root.glob(TACO_TRAIN_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no TACO train shards under {root}")
    columns = ["question", "starter_code", "input_output", "difficulty", "name"]
    if with_solutions:
        columns.append("solutions")
    index = 0
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        data = table.to_pylist()
        for row in data:
            yield {
                "taco_index": index,
                "question": row["question"],
                "solutions": parse_solutions(row["solutions"]) if with_solutions else [],
                "starter_code": row["starter_code"],
                "input_output": row["input_output"],
                "difficulty": row["difficulty"],
                "name": row["name"],
            }
            index += 1


def iter_taco_test(sources_dir: str | Path, with_solutions: bool = True) -> Iterator[dict]:
    """Yield TACO ALL/test rows in a stable order, tagged with their index.

    The index is the 0-based position in the concatenated test-shard order; the
    sibling task's held-out parquet encodes it as problem_id ``test<index>``
    (byte-verified positionally against its published test.parquet, the same
    way the train ``train<index>`` scheme was verified). The verified subset
    covers train rows only, so test-split cleanliness is established by this
    task's own purification run, not by likaixin/TACO-verified.
    """
    import pyarrow.parquet as pq

    root = Path(sources_dir)
    shards = sorted(root.glob(TACO_TEST_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no TACO test shards under {root}")
    columns = ["question", "starter_code", "input_output", "difficulty", "name"]
    if with_solutions:
        columns.append("solutions")
    index = 0
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        data = table.to_pylist()
        for row in data:
            yield {
                "test_index": index,
                "question": row["question"],
                "solutions": parse_solutions(row["solutions"]) if with_solutions else [],
                "starter_code": row["starter_code"],
                "input_output": row["input_output"],
                "difficulty": row["difficulty"],
                "name": row["name"],
            }
            index += 1


def iter_verified_records(path: str | Path, keys=("id",)) -> Iterator[dict]:
    """Stream the TACO-verified JSON array one record at a time.

    The file is a pretty-printed JSON array (records separated by whitespace
    and commas), so each boundary is found by skipping whitespace; each
    record is decoded whole (solutions included) but released one at a time,
    so peak memory is a single record, not the 1.9 GB file.
    """
    decoder = json.JSONDecoder()
    WS = " \t\n\r"
    with open(path, "r", encoding="utf-8") as fh:
        buf = ""
        pos = 0

        def skip_ws() -> bool:
            """Advance pos past whitespace, filling the buffer if exhausted.
            False at EOF."""
            nonlocal buf, pos
            while True:
                while pos < len(buf) and buf[pos] in WS:
                    pos += 1
                if pos < len(buf):
                    return True
                chunk = fh.read(1 << 22)
                if not chunk:
                    return False
                buf += chunk

        # find the opening bracket
        if not skip_ws():
            raise ValueError("taco_verified.json: empty file")
        if buf[pos] != "[":
            raise ValueError("taco_verified.json: not a JSON array")
        pos += 1

        records = 0
        while True:
            if not skip_ws():
                return
            if buf[pos] == "]":
                return
            # ensure we have enough of the next record to decode it
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    chunk = fh.read(1 << 22)
                    if not chunk:
                        raise
                    buf += chunk
            yield obj
            records += 1
            pos = end
            if not skip_ws():
                return
            if buf[pos] == "]":
                return
            if buf[pos] != ",":
                raise ValueError(
                    f"taco_verified.json: unexpected {buf[pos]!r} after record {records}"
                )
            pos += 1
            # keep the buffer bounded
            if pos > (1 << 22):
                buf = buf[pos:]
                pos = 0


def verified_id_set(path: str | Path) -> set[int]:
    ids = set()
    for rec in iter_verified_records(path):
        ids.add(int(rec["id"]))
    return ids


def verify_id_mapping(sources_dir: str | Path, verified_path: str | Path,
                      n_checks: int = 4, spread_every: int = 2000) -> list[str]:
    """Confirm that TACO-verified record ``id`` == the 0-based row index
    (``taco_index``) in the concatenated TACO ALL/train shard order.

    Verified empirically: the ids are sparse (only verified problems are in
    the file) and equal to taco_index, NOT taco_index+1 (60/60 sampled
    records across the whole file match by exact question text). Checks the
    first n_checks records plus one every ``spread_every`` records, by exact
    question text. Returns human-readable check results (also usable as
    receipts). Raises on mismatch -- a wrong mapping would silently corrupt
    the whole dirty/clean split.
    """
    checks = []
    needed = {}
    seen = 0
    for rec in iter_verified_records(verified_path):
        seen += 1
        vid = int(rec["id"])
        if seen <= n_checks or seen % spread_every == 0:
            needed[vid] = rec.get("question")
    if not needed:
        raise ValueError("no verified records checked")
    by_index = {}
    for row in iter_taco_train(sources_dir, with_solutions=False):
        if row["taco_index"] in needed:
            by_index[row["taco_index"]] = row["question"]
    for vid, q in sorted(needed.items()):
        tq = by_index.get(vid)
        ok = tq is not None and tq == q
        if not ok:
            raise ValueError(
                f"verified id={vid} does not match TACO ALL/train row "
                f"{vid} by question text; the id<->index mapping is NOT "
                "positional and every selection built on it is invalid"
            )
        checks.append(f"verified id={vid}: question match = {ok}")
    return checks


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
