"""Fetch pinned Hub assets and expose only the verified files to the task."""

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--dest", required=True, type=pathlib.Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (manifest["schema_version"] != 1 or manifest["repo_type"] not in ("model", "dataset")
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]) or not manifest["repo_id"]):
        raise ValueError("Invalid Hub asset manifest")
    files = {}
    for row in manifest["files"]:
        name = row["path"]
        path = pathlib.PurePosixPath(name)
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                or path in files or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
            raise ValueError(f"Invalid Hub asset entry: {name!r}")
        files[path] = row
    if not files:
        raise ValueError("Empty Hub asset manifest")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".asset-", dir=args.dest.parent) as temporary:
        staging = pathlib.Path(temporary)
        pinned = staging / "official.json"
        pinned.write_text(json.dumps({"files": {str(path): row["sha256"] for path, row in files.items()}}))
        downloaded = staging / "snapshot"
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).with_name("fetch_hf.py")),
             "--type", manifest["repo_type"], "--repo", manifest["repo_id"],
             "--revision", manifest["revision"], "--manifest", str(pinned), "--dest", str(downloaded)],
            check=True,
        )
        verified = staging / "verified"
        for path, row in files.items():
            source = downloaded / path
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if source.stat().st_size != row["size_bytes"] or digest != row["sha256"]:
                raise ValueError(f"Asset verification failed: {path}")
            target = verified / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.stat().st_mode & 0o004:
                source.rename(target)
            else:
                shutil.copyfile(source, target)
                target.chmod(0o644)
        verified.rename(args.dest)
    print(f"[prepare] verified {len(files)} file sizes and SHA-256 hashes")


if __name__ == "__main__":
    main()
