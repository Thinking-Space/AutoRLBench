#!/usr/bin/env python3
"""Fetch one pinned file by URL during the prepare stage.

Kept inside the task package, next to `prepare.sh` (invoked as
`python3 "$AUTORLBENCH_ENVIRONMENT/fetch_url.py" ...`), so the same CLI ships
with the task. Runs inside the evaluation image, which has no HF SDK --
standard library only.

Verifies byte size and SHA-256 before moving the file into place, so a
pinned-by-hash manifest stays authoritative even if the upstream repo moves.
An interrupted or mismatched fetch leaves no partial file behind.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

# hf-mirror 403s requests that send no User-Agent at all, so identify the
# client explicitly.
HEADERS = {"User-Agent": "task-fetch/1.0"}
# Cluster egress drops connections mid-transfer often enough that a single-shot
# download is unreliable: retry with backoff. A short read (the mirror serves
# 200-status throttle pages) is discarded rather than resumed, since its bytes
# are garbage; a transfer that appended nothing resumes via a Range request.
RETRIES = 8


def fetch(url: str, sha256: str, size: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    for attempt in range(RETRIES):
        done = partial.stat().st_size if partial.exists() else 0
        headers = dict(HEADERS)
        if done:
            headers["Range"] = f"bytes={done}-"
        exc: OSError | None = None
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                with partial.open("ab" if done else "wb") as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
        except OSError as caught:
            exc = caught
        got = partial.stat().st_size if partial.exists() else 0
        if got >= size:
            break
        if got != done:
            partial.unlink(missing_ok=True)
        if attempt + 1 == RETRIES:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"download failed for {url} after {RETRIES} attempts: "
                f"{exc or f'short read {got}/{size}'}"
            )
        wait = min(10 * 2**attempt, 120)
        print(
            f"[fetch] retry {attempt + 1} after {exc or 'short read'}; "
            f"sleeping {wait}s",
            flush=True,
        )
        time.sleep(wait)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if partial.stat().st_size != size:
        partial.unlink()
        raise RuntimeError(f"size mismatch for {url}: expected {size}, got {partial.stat().st_size}")
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(
            f"sha256 mismatch for {url}: expected {sha256}, "
            f"got {digest.hexdigest()}"
        )
    partial.replace(dest)
    print(f"[fetch] {url} -> {dest} ({size} bytes)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fetch_url")
    parser.add_argument(
        "--type",
        choices=("model", "dataset"),
        default="dataset",
        help="kept for CLI parity with the fetch helpers; this tool is URL-based",
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fetch(args.url, args.sha256, args.size, args.dest)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
