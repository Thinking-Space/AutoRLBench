#!/usr/bin/env python3
"""Prepare the recorded logical rows and, when requested, the pinned model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ENVIRONMENT = Path(__file__).resolve().parent
ROW_COUNTS = {"train": 1280, "val": 24}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_rows(workspace: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = workspace / "data_evidence"
    expected = json.loads((source / "checksums.json").read_text())
    checksums = {}
    for split in ROW_COUNTS:
        name = f"{split}.jsonl"
        record = expected[name]
        checksums[name] = record["sha256"] if isinstance(record, dict) else record
        if sha256(source / name) != checksums[name]:
            raise ValueError(f"Recorded input bytes changed: {name}")

    dest = workspace / "data"
    dest.mkdir(exist_ok=True)
    marker = dest / ".complete"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous["source_sha256"] == checksums and all(
            (dest / name).is_file() and sha256(dest / name) == digest
            for name, digest in previous["output_sha256"].items()
        ):
            print(f"Using verified Parquet data in {dest}")
            return

    with tempfile.TemporaryDirectory(prefix=".inventory-stage-", dir=dest) as temporary:
        stage = Path(temporary)
        manifest = {"outputs": {}}
        files = {}
        for split, expected_rows in ROW_COUNTS.items():
            with (source / f"{split}.jsonl").open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != expected_rows:
                raise ValueError(f"Expected {expected_rows} {split} rows, got {len(rows)}")
            path = stage / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
            files[path.name] = sha256(path)
            manifest["outputs"][split] = {
                "path": path.name, "rows": len(rows), "sha256": files[path.name],
                "size_bytes": path.stat().st_size,
            }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        files[manifest_path.name] = sha256(manifest_path)
        for name in files:
            path = stage / name
            path.chmod(0o444)
            path.replace(dest / name)
        receipt = stage / ".complete"
        receipt.write_text(json.dumps({"source_sha256": checksums, "output_sha256": files}, indent=2) + "\n")
        receipt.replace(marker)
    print(f"Prepared {ROW_COUNTS['train']} training and {ROW_COUNTS['val']} validation rows in {dest}")


def verify_model(directory: Path, asset: dict) -> None:
    for record in asset["files"]:
        path = directory / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed pinned model file: {path}")


def prepare_model(workspace: Path, *, cache_only: bool) -> None:
    # The verbatim AutoRLBench helper supplies shared cache locations and
    # portable hard-link/copy materialization. Pin the revision and file bytes here.
    import fcntl
    from fetch_hf import _force_disable_xet, materialize, repo_cache_dir

    asset = json.loads((ENVIRONMENT / "asset_registry.json").read_text())["model"]
    destination = workspace / "model"
    if destination.exists():
        verify_model(destination, asset)
        print(f"Verified model in {destination}")
        return

    repo, revision = asset["repo"], asset["revision"]
    shared = repo_cache_dir(repo, "model")
    pinned = repo_cache_dir(repo, "model", f"{repo}@{revision}")
    for cache in (pinned, shared):
        if cache.is_dir():
            try:
                verify_model(cache, asset)
            except ValueError:
                continue
            materialize(cache, destination)
            print(f"Prepared model from verified cache {cache}")
            return

    if cache_only:
        raise FileNotFoundError(f"No verified model cache found in {pinned} or {shared}")

    from huggingface_hub import snapshot_download

    _force_disable_xet()
    pinned.parent.mkdir(parents=True, exist_ok=True)
    with pinned.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not pinned.exists():
            with tempfile.TemporaryDirectory(prefix=".inventory-model-", dir=pinned.parent) as temporary:
                stage = Path(temporary) / "snapshot"
                snapshot_download(repo_id=repo, revision=revision, local_dir=stage,
                                  allow_patterns=[record["path"] for record in asset["files"]])
                verify_model(stage, asset)
                stage.replace(pinned)
        else:
            verify_model(pinned, asset)
        materialize(pinned, destination)
    print(f"Prepared {repo}@{revision} in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="prepare the model instead of the data")
    parser.add_argument("--cache-only", action="store_true", help="verify local model bytes or fail without downloading")
    args = parser.parse_args()
    workspace = Path(os.environ["AUTORLBENCH_WORKSPACE"]).resolve()
    if args.model:
        prepare_model(workspace, cache_only=args.cache_only)
    else:
        if args.cache_only:
            parser.error("--cache-only requires --model")
        prepare_rows(workspace)


if __name__ == "__main__":
    main()
