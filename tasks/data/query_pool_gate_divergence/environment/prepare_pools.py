"""Materialise the query pools the case lists refer to.

The pools are not shipped; the per-source query banks are, and this script rebuilds every pool
from them with `scripts/build_pool.py` and the recipes in `pool_recipes.json`, then derives the
manifest variants, then checks every resulting file against the digests in `pool_digests.json`.
A pool whose bytes do not match its recorded digest aborts prepare: the case lists pin the same
digests, so a pool that differs here would simply be rejected later for the wrong reason.

usage: prepare_pools.py --environment <dir> --workspace <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(pool: Path) -> dict:
    return json.loads((pool / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(pool: Path, manifest: dict) -> None:
    (pool / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def unpack_banks(environment: Path, banks_dir: Path) -> None:
    archive = environment / "sources" / "query_banks.tar.xz"
    expected = json.loads((environment / "sources_manifest.json").read_text())["query_banks_sha256"]
    actual = sha256(archive)
    if actual != expected:
        raise SystemExit(f"prepare: query_banks.tar.xz digest mismatch: {actual} != {expected}")
    if banks_dir.exists():
        shutil.rmtree(banks_dir)
    banks_dir.mkdir(parents=True)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(banks_dir)


def build(workspace: Path, banks: Path, registry: Path, recipe: dict, pools: Path) -> None:
    command = [
        sys.executable, "-B", str(workspace / "scripts" / "build_pool.py"),
        "--banks", str(banks),
        "--out", str(pools / recipe["pool_id"]),
        "--total", str(recipe["total"]),
        "--weights", recipe["weights"],
        "--registry", str(registry),
        "--seed", str(recipe["seed"]),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"prepare: building {recipe['pool_id']} failed\n{completed.stdout[-1500:]}")


def derive(pools: Path, recipe: dict) -> None:
    target = pools / recipe["pool_id"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(pools / recipe["from"], target)
    manifest = load_manifest(target)
    edit = recipe["edit"]
    if edit == "drop_zero_sources":
        manifest["source_counts"] = {k: v for k, v in manifest["source_counts"].items() if v}
        manifest["excluded_instance_counts"] = {
            k: v for k, v in manifest["excluded_instance_counts"].items()
            if k in manifest["source_counts"]
        }
    elif edit == "total_plus":
        manifest["total_queries"] = manifest["total_queries"] + recipe["amount"]
    elif edit == "drop_sha256":
        manifest.pop("sha256", None)
    elif edit == "total_as_string":
        manifest["total_queries"] = str(manifest["total_queries"])
    elif edit == "drop_total":
        manifest.pop("total_queries", None)
    elif edit == "truncate_rows":
        lines = (target / "queries.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (target / "queries.jsonl").write_text("".join(lines[: -recipe["amount"]]), encoding="utf-8")
    else:
        raise SystemExit(f"prepare: unknown edit {edit!r}")
    manifest["note"] = recipe["note"]
    write_manifest(target, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    environment = Path(args.environment).resolve()
    workspace = Path(args.workspace).resolve()

    recipes = json.loads((environment / "pool_recipes.json").read_text(encoding="utf-8"))
    digests = json.loads((environment / "pool_digests.json").read_text(encoding="utf-8"))

    banks = workspace / ".banks"
    unpack_banks(environment, banks)
    pools = workspace / "pools"
    if pools.exists():
        shutil.rmtree(pools)
    pools.mkdir(parents=True)
    registry = workspace / "registry" / "missing_images.json"

    for recipe in recipes["built"]:
        build(workspace, banks, registry, recipe, pools)
    for recipe in recipes["derived"]:
        derive(pools, recipe)

    mismatched = []
    for pool_id, files in sorted(digests.items()):
        for name, expected in sorted(files.items()):
            path = pools / pool_id / name
            actual = sha256(path) if path.is_file() else None
            if actual != expected:
                mismatched.append(f"{pool_id}/{name}: {actual} != {expected}")
    if mismatched:
        raise SystemExit("prepare: rebuilt pools do not match their recorded digests:\n  "
                         + "\n  ".join(mismatched[:10]))

    shutil.rmtree(banks)
    print(f"prepared {len(digests)} query pools, every file matching its recorded digest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
