"""Prepare an isolated, explicitly labeled synthetic data-loading fault store."""

import argparse
import json
from pathlib import Path

if __package__:
    from .storage import prepare
else:
    from storage import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-group", type=int, default=32)
    parser.add_argument("--eval-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)
    try:
        marker = prepare(args.source, args.out, args.train_per_group, args.eval_per_group, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(marker, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
