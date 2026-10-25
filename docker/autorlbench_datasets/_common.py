"""Shared helpers: write parquet and print check arguments ready to paste into prepare.sh."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


def hf_cache_root() -> Path:
    """Cache root for the HF libraries; on the host this is datasets/hf.

    The layout deliberately matches what `HF_HOME=$AUTORLBENCH_DATASETS/hf` would
    produce (hub in hf/hub, datasets in hf/datasets), so it is the same tree that
    autorlbench_fetch.hf_cache_dir() looks at.

    **Must be called before importing huggingface_hub / datasets.** Both libraries
    read HF_HOME into module-level constants (constants.HF_HUB_CACHE) at import
    time, so changing os.environ afterwards is a silent no-op. That is exactly how
    autorlbench_fetch.py once downloaded 954MB of weights into the container's
    writable layer. The recipes defer their imports into functions so this call can
    run first; hoisting those imports to module level breaks that guarantee.
    """
    root = Path(os.environ.get("AUTORLBENCH_DATASETS", "/autorlbench/datasets")) / "hf"
    root.mkdir(parents=True, exist_ok=True)
    # Besides the explicit cache_dir passed below, some paths (metadata, etags,
    # the hf_hub_download calls inside load_dataset) only honor the env var, so
    # set both.
    os.environ.setdefault("HF_HOME", str(root))
    return root


def hf_datasets_cache() -> Path:
    """The directory to pass as `load_dataset(cache_dir=...)`."""
    return hf_cache_root() / "datasets"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(rows: list[dict], out_dir: Path, name: str) -> Path:
    """Write one split and return its path.

    Goes through pandas -> parquet explicitly instead of datasets' own to_parquet:
    the latter's sharding and compression defaults change between versions, and we
    need "same input, same bytes".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def report(paths: list[Path]) -> None:
    """Print check arguments ready to paste into prepare.sh, so nobody copies them by hand.

    `autorlbench_dataset --expect/--expect-size` share semantics with the same-named
    flags of `autorlbench_fetch_hf`: only files the task explicitly declares are
    checked, and a mismatch is a hard failure. The recipe already asserts row counts
    and schema, so these flags are an **optional** second gate: they catch "the
    cached copy was modified", which row-count assertions cannot.
    """
    print()
    print("=" * 72)
    print("Append the following to the autorlbench_dataset call in the task's environment/prepare.sh:")
    print("=" * 72)
    for p in sorted(paths):
        print(f"    --expect {p.name}={sha256_of(p)} \\")
        print(f"    --expect-size {p.name}={p.stat().st_size} \\")

