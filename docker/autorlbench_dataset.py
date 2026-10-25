#!/usr/bin/env python3
"""Build a task's dataset from its upstream source and a deterministic recipe.

Installed into the image as `autorlbench_dataset`, and called by each task's
`environment/prepare.sh` for the datasets that are *derived* rather than
downloaded:

    autorlbench_dataset --recipe dapo_math_17k \\
                     --dest "$AUTORLBENCH_WORKSPACE/DAPO-Math-17k_verl"

`autorlbench_fetch_hf` covers the other case -- a repo whose bytes are already what
the task needs. These datasets are not: upstream `openai/gsm8k` is two raw
columns and upstream `DAPO-Math-17k` is a single unsplit shard, and verl's
`RLHFDataset` accepts neither. The recipe in `autorlbench_datasets/<recipe>.py` is
the transformation, and it is the thing that has to stay fixed, not the
download. The package is called `autorlbench_datasets` and not `datasets` because
the recipes import HuggingFace's `datasets`, and a sibling package by that name
on sys.path would shadow it.

Cache-first, for the same reason `autorlbench_fetch_hf` is: 011 / 022 / 023 all
want the same `DAPO-Math-17k_verl`, and rebuilding it per task would download
17917 rows three times to produce three identical trees.

The cache key is `<recipe>-v<RECIPE_VERSION>`, not a hash of the recipe source.
A source hash would look more rigorous and be worse: reflowing a comment would
silently orphan every cached copy, while the version constant is a statement by
whoever edited the recipe about whether the *output* changed. Editing the split
rule without bumping it is the one mistake this cannot catch, which is why the
constant sits next to the rule with a comment saying so.

Like `autorlbench_fetch.py` this is deliberately standalone -- it runs in the
task's environment image, which has `datasets`/`pandas`/`pyarrow` but not the
harness.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    # `autorlbench_fetch.py` is the sibling file in both layouts that matter: the
    # repository's docker/ directory and /usr/local/libexec in the image.
    sys.path.insert(0, str(_HERE))

from autorlbench_fetch import FetchError, materialize, require_verified  # noqa: E402

# Cache root for derived datasets. Same directory as autorlbench_fetch's
# DATASETS_ROOT: downloaded and locally built data are both datasets, so there
# is no reason to split them.
DATASETS_ROOT = Path(os.environ.get("AUTORLBENCH_DATASETS", "/autorlbench/datasets"))

# The recipes a task may name. An explicit list, not a directory scan: a typo
# should be an error naming the valid choices, not an import of whatever
# happens to be on disk.
RECIPES = ("dapo_math_17k", "gsm8k", "search_r1_nq_hotpotqa", "wiki18_e5")

# Written into a finished cache entry, and checked before one is reused. A
# recipe that dies partway through leaves real .parquet files behind, and
# without this marker the next run would treat that half-built tree as a hit and
# train on a truncated dataset.
MARKER = ".autorlbench-complete"


def load_recipe(name: str):
    try:
        return importlib.import_module(f"autorlbench_datasets.{name}")
    except ImportError as exc:  # pragma: no cover - import failure is fatal anyway
        raise FetchError(f"cannot load recipe {name!r}: {exc}") from exc


def build_cached(name: str, *, allow_count_mismatch: bool) -> Path:
    """Return a finished cache entry for `name`, building it if it is not there."""
    module = load_recipe(name)
    version = getattr(module, "RECIPE_VERSION", None)
    if not version:
        raise FetchError(f"recipe {name!r} has no RECIPE_VERSION; it cannot be cached safely")

    # "derived" sits next to hf/: one holds downloads, the other recipe output,
    # as two trees under the same datasets root.
    entry = DATASETS_ROOT / "derived" / f"{name}-v{version}"
    if (entry / MARKER).is_file():
        print(f"[dataset] cache hit  {name} v{version} -> {entry}")
        return entry

    # A leftover without the marker is a failed build, not data. Drop it rather
    # than building on top of it.
    if entry.exists():
        print(f"[dataset] discarding incomplete cache entry {entry}")
        shutil.rmtree(entry)

    staging = entry.with_name(entry.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"[dataset] building {name} v{version}")
    try:
        module.build(staging, allow_count_mismatch=allow_count_mismatch)
        (staging / MARKER).write_text(f"{name} v{version}\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Rename last, so the marker and the data become visible together.
    os.replace(staging, entry)
    return entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autorlbench_dataset",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--dest", required=True, type=Path, help="where the task expects it")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "continue when upstream's row count no longer matches what the recipe "
            "recorded. Off by default: a changed row count means the grader's "
            "baseline is measured against a different dataset"
        ),
    )
    parser.add_argument(
        "--expect",
        action="append",
        metavar="PATH=SHA256",
        help="declared checksum, relative to the built tree; repeatable",
    )
    parser.add_argument(
        "--expect-size",
        action="append",
        metavar="PATH=BYTES",
        help="declared size, relative to the built tree; repeatable",
    )
    args = parser.parse_args(argv)

    try:
        entry = build_cached(args.recipe, allow_count_mismatch=args.allow_count_mismatch)
        expect = dict(_pairs(args.expect, "--expect"))
        sizes = {key: int(value) for key, value in _pairs(args.expect_size, "--expect-size")}
        # Only checks what the task declared; an entry with neither is checked
        # by the recipe's own row-count and schema assertions instead.
        require_verified(entry, expect, sizes, f"{args.recipe} dataset")
        materialize(entry, args.dest)
        # The marker is cache bookkeeping. Leaving it in the workspace would put
        # a stray dotfile in front of whatever globs the training script does.
        (args.dest / MARKER).unlink(missing_ok=True)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"[dataset] ready {args.dest}")
    return 0


def _pairs(items: Optional[Sequence[str]], flag: str):
    for item in items or []:
        relative, separator, value = item.partition("=")
        if not relative or not separator or not value:
            raise FetchError(f"{flag} wants `<relative-path>=<value>`, got {item!r}")
        yield relative, value


if __name__ == "__main__":
    raise SystemExit(main())
