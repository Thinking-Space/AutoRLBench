"""PeterJinGo/wiki-18-e5-index + wiki-18-corpus -> wiki-18/{e5_Flat.index,wiki-18.jsonl}

Shared by tasks/slime_search_r1 and tasks/slime_search_r1_nomask: the recipe's retrieval
service needs it (examples/search-r1/start_retriver.sh -> local_dense_retriever/
retrieval_server.py), and the retrieval section of run_qwen2.5_3B.sh (from line 96) starts
that service itself when nothing is listening on :8000, waits for it to be ready, and exits
if it never is.

## Why a "recipe" and not autorlbench_fetch_hf

Neither upstream asset is usable as downloaded:

  * the index is split into two ~40GB `part_*` files that nothing can read until they are `cat`ed into one;
  * the corpus **is named `wiki-18.jsonl.gz` but is actually a tar**.

The second is a trap worth writing down: it is a single-member GNU tar.gz whose member name
still carries a path from the machine that packed it (see CORPUS_MEMBER). Unpacking it with
upstream's `gzip -df` (prepare_data.sh:110-113) yields the tar archive itself: the first 100
bytes are the member name, and from byte 124 is the base-256 encoded member size, first byte
0x80. That file is named wiki-18.jsonl, and when the retrieval service reads it as a json
dataset it dies with `UnicodeDecodeError: invalid start byte 0x80`, far from the cause. So
this extracts the member with tarfile and pins both the member name and size: if upstream
ever repacks, this must stop immediately rather than produce an unreadable corpus.

The index assembly rule is still copied from examples/search-r1/prepare_data.sh:100-113.

## Size: the largest asset in this repo

    part_aa            42,949,672,960 B   (40 GiB)
    part_ab            21,609,402,413 B   (20 GiB)
    e5_Flat.index      64,559,075,373 B   (assembled, 60 GiB)
    wiki-18.jsonl.gz    5,123,307,260 B   (actually a tar.gz)
    wiki-18.jsonl      14,393,573,105 B   (extracted)

Download one part, append it, delete it right away, so peak usage is ~86GB rather than 150GB.
The finished cache entry is ~79GB, and each run gets it by hardlink (materialize) without
another copy, provided $AUTORLBENCH_DATASETS and the run directory are on the same filesystem.
If not, autorlbench_dataset falls back to a real copy, 79GB each time; keep an eye on that.

## Failure midway: outputs are built outside staging

autorlbench_dataset retries by "no .autorlbench-complete marker means delete the whole tree and
start over". That is fine for parquet recipes that finish in minutes, but not for this one: an
error in the corpus step would also throw away the already downloaded and assembled 60GiB index,
and download 70GB again.

So both outputs are first built in $AUTORLBENCH_DATASETS/scratch/wiki18_e5, and only once both
are complete are they `shutil.move`d into staging (same datasets root, so a single rename). On
failure scratch is **deliberately kept**, and the next run skips assembly when it finds an
e5_Flat.index of the right size. The cost is tens of GB left on disk after a failure; the path
is printed, delete it yourself if unwanted.

A half-assembled index is never taken as finished: skipping requires an exact byte count, and
one byte off means reassembling. This must not be loosened: faiss does not reject a truncated
index, it just returns wrong neighbors.

## Verification

Size only, no sha256: upstream publishes no hash for the assembled file, hashing 64GB costs over
ten minutes per build, and what it would catch is truncation, which the size already shows. The
parts themselves are pinned by revision, and hf_hub_download verifies the download.

Usage:
    python3 -m autorlbench_datasets.wiki18_e5 --out-dir ./prepared/wiki-18
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from ._common import hf_cache_root

REPO_INDEX = "PeterJinGo/wiki-18-e5-index"
REVISION_INDEX = "a4d31160a035f30764604f4827cd8f1d0315eb86"
# Order is the shell glob order of `cat part_*`; reversed, faiss still reads a header
# and then returns completely wrong retrieval results.
PARTS = (("part_aa", 42949672960), ("part_ab", 21609402413))
INDEX_NAME = "e5_Flat.index"
INDEX_BYTES = sum(size for _, size in PARTS)

REPO_CORPUS = "PeterJinGo/wiki-18-corpus"
REVISION_CORPUS = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_GZ = "wiki-18.jsonl.gz"
CORPUS_GZ_BYTES = 5123307260
CORPUS_NAME = "wiki-18.jsonl"
# Name of the archive's only member. A leftover path from the packing machine, not a typo; see the docstring.
CORPUS_MEMBER = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
CORPUS_BYTES = 14393573105

# retrieval_server.py:34-41 reads this jsonl as a datasets json dataset,
# and generate_with_search.py:51 reads document["contents"].
CORPUS_KEYS = ("id", "contents")

# Part of the cache key (see docker/autorlbench_dataset.py). Bump it whenever the recipe changes.
# Still 1: v1 never produced an entry with .autorlbench-complete (the tar trap above killed it at
# the corpus step every time), so there is no old cache to invalidate.
RECIPE_VERSION = "1"

_CHUNK = 32 << 20


def scratch_dir(out_dir: Path) -> Path:
    """Working directory for download and assembly, **outside staging** (see the docstring for why).

    A manual run on the host (`python3 -m autorlbench_datasets.wiki18_e5`) has no
    AUTORLBENCH_DATASETS, so fall back to under out_dir; that usage picks its own directory anyway.
    """
    root = os.environ.get("AUTORLBENCH_DATASETS")
    base = Path(root) if root else out_dir
    return base / "scratch" / "wiki18_e5"


def _done(path: Path, size: int) -> bool:
    """Whether this step's output is already there and complete.

    Only an exact byte count counts. ">= size" or "exists" would both let a previous truncated result through.
    """
    return path.is_file() and path.stat().st_size == size


def _download(repo: str, revision: str, filename: str, expected: int, into: Path) -> Path:
    # Before the import: see _common.hf_cache_root(). The 40GB payload lands on the host via
    # local_dir below; this covers huggingface_hub's own metadata/etag cache.
    hf_cache_root()
    from huggingface_hub import hf_hub_download

    print(f"downloading {repo}@{revision[:12]} {filename} ({expected / 1e9:.1f} GB) ...", flush=True)
    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            # Write a plain file directly instead of keeping another 40GB in the blob cache. Resume
            # bookkeeping (.cache/huggingface/download/*.incomplete) also stays here, so a rerun
            # resumes instead of starting over.
            local_dir=str(into),
        )
    )
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(
            f"{filename} downloaded as {actual} bytes, the recipe expects {expected}. "
            f"Upstream changed or the download was truncated."
        )
    return path


def build_index(scratch: Path) -> Path:
    """Assemble the index the way prepare_data.sh does: `cat part_* > e5_Flat.index`."""
    index = scratch / INDEX_NAME
    if _done(index, INDEX_BYTES):
        print(f"  {INDEX_NAME} is already in {scratch} with the right size, skipping assembly", flush=True)
        return index

    with index.open("wb") as sink:
        for name, size in PARTS:
            part = _download(REPO_INDEX, REVISION_INDEX, name, size, scratch)
            print(f"  appending {name} -> {index.name}", flush=True)
            with part.open("rb") as source:
                shutil.copyfileobj(source, sink, _CHUNK)
            # Delete right after appending: the two parts total 60GiB, and keeping them doubles peak usage.
            part.unlink()
    actual = index.stat().st_size
    if actual != INDEX_BYTES:
        raise SystemExit(f"{INDEX_NAME} assembled to {actual} bytes, expected {INDEX_BYTES}.")
    print(f"  -> {index} ({actual} bytes)", flush=True)
    return index


def _check_first_line(corpus: Path) -> None:
    with corpus.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        raise SystemExit(f"first line of {CORPUS_NAME} is empty")
    row = json.loads(first)
    missing = [key for key in CORPUS_KEYS if key not in row]
    if missing:
        raise SystemExit(
            f"first line of {CORPUS_NAME} is missing fields {missing}; the retrieval service reads"
            f" documents by {list(CORPUS_KEYS)}"
            f" (retrieval_server.py:39-41 + generate_with_search.py:51)"
        )


def build_corpus(scratch: Path) -> Path:
    """Extract the jsonl member from wiki-18.jsonl.gz.

    The upstream .gz is a tar.gz, not a gzipped jsonl; see the module docstring.
    """
    corpus = scratch / CORPUS_NAME
    if _done(corpus, CORPUS_BYTES):
        print(f"  {CORPUS_NAME} is already in {scratch} with the right size, skipping extraction", flush=True)
        _check_first_line(corpus)
        return corpus

    gz = _download(REPO_CORPUS, REVISION_CORPUS, CORPUS_GZ, CORPUS_GZ_BYTES, scratch)
    print(f"  extracting {CORPUS_MEMBER} from {CORPUS_GZ} -> {corpus.name}", flush=True)
    # `r|gz` streams: one pass, decompressing as it writes. `r:gz` would scan all 14GB once
    # to build the member table, then extractfile would make a second pass.
    with tarfile.open(gz, "r|gz") as archive:
        member = archive.next()
        while member is not None and not member.isfile():
            member = archive.next()
        if member is None:
            raise SystemExit(f"{CORPUS_GZ} has no regular file member")
        if member.name != CORPUS_MEMBER:
            raise SystemExit(
                f"the member in {CORPUS_GZ} is {member.name!r}, the recipe expects {CORPUS_MEMBER!r}. "
                f"Upstream repacked it; check the contents before updating the recipe."
            )
        if member.size != CORPUS_BYTES:
            raise SystemExit(
                f"{CORPUS_MEMBER} is declared as {member.size} bytes in the archive, "
                f"the recipe expects {CORPUS_BYTES}."
            )
        source = archive.extractfile(member)
        with corpus.open("wb") as sink:
            shutil.copyfileobj(source, sink, _CHUNK)
    gz.unlink()

    actual = corpus.stat().st_size
    if actual != CORPUS_BYTES:
        raise SystemExit(f"{CORPUS_NAME} extracted to {actual} bytes, expected {CORPUS_BYTES}.")
    _check_first_line(corpus)
    print(f"  -> {corpus} ({actual} bytes)", flush=True)
    return corpus


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """Assemble the index and extract the corpus, moving both into out_dir once both are done;
    return the written paths.

    `allow_count_mismatch` has no use in this recipe (there are no "row counts" here; it checks
    byte counts, and a byte mismatch means a broken asset, with no room for "approve and continue").
    The parameter stays because autorlbench_dataset calls every recipe with the same signature.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir(out_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[wiki18_e5] working directory {scratch} (kept on failure for the next run to reuse)", flush=True)

    built = [build_index(scratch), build_corpus(scratch)]

    written = []
    for source in built:
        target = out_dir / source.name
        # Under the same datasets root, move is a single rename; a cross-device copy fallback
        # is still correct.
        shutil.move(str(source), str(target))
        written.append(target)

    # Both outputs have been moved by now; only hf_hub_download's bookkeeping directory is left.
    shutil.rmtree(scratch, ignore_errors=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    written = build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)
    print()
    for path in written:
        print(f"    --expect-size {path.name}={path.stat().st_size} \\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
