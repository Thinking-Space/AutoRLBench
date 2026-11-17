#!/usr/bin/env python3
"""Download one URL and fail loudly unless its byte size and SHA-256 match."""
import argparse
import hashlib
import os
import sys
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--size", required=True, type=int)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()

    digest, seen = hashlib.sha256(), 0
    tmp = a.dest + ".part"
    with urllib.request.urlopen(a.url) as resp, open(tmp, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            digest.update(block)
            seen += len(block)
            out.write(block)
    if seen != a.size or digest.hexdigest() != a.sha256:
        os.remove(tmp)
        sys.exit(f"{a.url}: got {seen} bytes, sha256 {digest.hexdigest()}; "
                 f"expected {a.size} bytes, sha256 {a.sha256}")
    os.replace(tmp, a.dest)


if __name__ == "__main__":
    main()
