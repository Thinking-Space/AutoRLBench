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

    # Local mirror escape hatch (audit addition): when the exact bytes are
    # already staged at $AUTORLBENCH_URL_MIRROR/<basename>, use them after
    # verifying the same pins — shared eval clusters cannot reach some hosts
    # (e.g. github archives over TLS-proxied git) that the author's own
    # environment could.
    import re as _re
    _basename = _re.sub(r"[^A-Za-z0-9._-]", "_", a.url.rsplit("/", 1)[-1] or "file")
    mirror_dir = os.environ.get("AUTORLBENCH_URL_MIRROR", "")
    if mirror_dir:
        candidate = os.path.join(mirror_dir, _basename)
        if os.path.isfile(candidate):
            _digest = hashlib.sha256()
            _seen = 0
            with open(candidate, "rb") as f:
                while True:
                    block = f.read(1 << 20)
                    if not block:
                        break
                    _digest.update(block)
                    _seen += len(block)
            if _seen == a.size and _digest.hexdigest() == a.sha256:
                import shutil as _shutil
                _shutil.copyfile(candidate, a.dest)
                print(f"fetch_url: local mirror hit for {a.url}")
                return
            print(f"fetch_url: mirror candidate {candidate} failed pins; "
                  f"falling back to network", file=sys.stderr)

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
