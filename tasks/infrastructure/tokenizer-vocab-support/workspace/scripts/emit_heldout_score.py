"""Emit the combined native #71 held-out score for one completed evaluation."""

import argparse
from decimal import Decimal
from pathlib import Path
from runpy import run_path
import sys

from summarize import DRAWS, heldout, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    args = parser.parse_args()

    expected = inputs(args.workspace / "data")
    sys.path.insert(0, str(args.workspace.resolve() / "framework"))
    score = run_path(str(args.workspace / "reward.py"))["compute_score"]
    counts, digest = heldout(args.validation, expected, args.step, native_score=score)
    correct = sum(counts.values())
    total = len(expected) * DRAWS
    print(f"Verified {correct}/{total} native answers, sha256={digest}")
    print(f"step:{args.step} - val-core/overall/acc/mean@32:{Decimal(correct) / Decimal(total)}")


if __name__ == "__main__":
    main()
