"""Revision-aware entry point around the shared cache materializer."""
import argparse
import functools
import os
import re

import fetch_hf


def main():
    parser = fetch_hf.build_parser()
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('revision must be a full commit hash')
    # Cache identity includes the immutable source version.
    args.cache_as = (args.cache_as or args.repo) + '-' + args.revision
    import huggingface_hub
    huggingface_hub.snapshot_download = functools.partial(
        huggingface_hub.snapshot_download, revision=args.revision)
    fetch_hf.fetch_hf(args)


if __name__ == '__main__':
    main()
