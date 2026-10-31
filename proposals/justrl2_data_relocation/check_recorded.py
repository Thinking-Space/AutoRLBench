#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the shipped audit archive and independently recompute all three arms."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from verify import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    archive = here / 'evidence/recorded-audit.tar.gz'
    pins = json.loads((here / 'evidence/sha256.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != pins['recorded-audit.tar.gz']:
        raise ValueError('Recorded evidence archive SHA256 mismatch')
    with tempfile.TemporaryDirectory(prefix='relocation-recorded-') as temporary:
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter='data')
        root = Path(temporary) / 'recorded'
        files = json.loads((root / 'sha256.json').read_text())
        for name, digest in files.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Recorded file SHA256 mismatch: ' + name)
        result = compare_runs(root / 'healthy', root / 'faulty', root / 'repaired')
        brief = {key: value for key, value in result.items() if key != 'runs'}
        brief['runs'] = {arm: {key: run[key] for key in ['pre', 'post', 'eval', 'optimizer', 'provenance']}
                         for arm, run in result['runs'].items()}
        if args.out:
            args.out.write_text(json.dumps(brief, indent=2) + '\n')
        print(json.dumps({key: value for key, value in brief.items() if key != 'runs'}, indent=2))
        return 0 if result['causal_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
