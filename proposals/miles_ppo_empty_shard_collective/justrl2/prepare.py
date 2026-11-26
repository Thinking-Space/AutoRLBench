#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download pinned public assets, create the pair and construct the replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent


def file_sha(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def fetch(asset, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == asset["size"] and file_sha(dest) == asset["sha256"]:
        return
    temp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(asset["url"], timeout=120) as response, temp.open("wb") as output:
        while block := response.read(8 * 1024**2):
            output.write(block)
    if temp.stat().st_size != asset["size"] or file_sha(temp) != asset["sha256"]:
        raise RuntimeError("Pinned asset verification failed: " + dest.name)
    temp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-only", action="store_true", help="Offline construction checks; no usable training weights")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "assets.json").read_text())
    fetch(manifest["public_source_archive"], root / "source.tar.gz")
    source_root = root / "source"
    source = source_root / ("JustRL-II-" + manifest["justrl2_revision"])
    if not source.exists():
        with tarfile.open(root / "source.tar.gz") as tar:
            for member in tar.getmembers():
                if Path(member.name).is_absolute() or ".." in Path(member.name).parts or member.issym() or member.islnk():
                    raise ValueError("Unexpected archive path or link")
            tar.extractall(source_root, filter="data")
    for asset in manifest["assets"]:
        if args.tokenizer_only and asset["path"].endswith(".bin"):
            continue
        fetch(asset, root / asset["path"])
    subprocess.run([sys.executable, str(HERE / "build_case.py"), "--source", str(source),
                    "--output", str(root / "pair")], check=True)
    subprocess.run([sys.executable, str(HERE / "make_replay.py"), "--source", str(source),
                    "--model", str(root / "model"), "--output", str(root / "replay")], check=True)
    print("Public case prepared; native weight conversion and GPU training remain separate validation steps.")


if __name__ == "__main__":
    main()
