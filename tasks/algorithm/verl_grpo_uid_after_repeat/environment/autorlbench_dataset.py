#!/usr/bin/env python3
"""Build a task's dataset from its upstream source and a deterministic recipe.

Lives in a task's `environment/` next to `prepare.sh` (alongside
`fetch_hf.py` and the `autorlbench_datasets/` package), for the datasets that
are *derived* rather than downloaded. prepare.sh runs it directly:

    python3 "$AUTORLBENCH_ENVIRONMENT/autorlbench_dataset.py" \\
        --recipe dapo_math_17k \\
        --dest "$AUTORLBENCH_WORKSPACE/DAPO-Math-17k_verl"

`fetch_hf.py` covers the other case -- a repo whose bytes are already what the
task needs. These datasets are not: upstream `DAPO-Math-17k` is a single
unsplit shard and the Search-R1 corpora need splitting/stitching, and verl's
`RLHFDataset` accepts neither. The recipe in `autorlbench_datasets/<recipe>.py`
is the transformation, and it is the thing that has to stay fixed, not the
download. The package is called `autorlbench_datasets` and not `datasets`
because the recipes import HuggingFace's `datasets`, and a sibling package by
that name on sys.path would shadow it.

Cache-first: the per-recipe build lands in
`$AUTORLBENCH_DATASETS/derived/<recipe>-v<RECIPE_VERSION>` and is reused across
prepare runs rather than rebuilt each time.

The cache key is `<recipe>-v<RECIPE_VERSION>`, not a hash of the recipe source.
A source hash would look more rigorous and be worse: reflowing a comment would
silently orphan every cached copy, while the version constant is a statement by
whoever edited the recipe about whether the *output* changed. Editing the split
rule without bumping it is the one mistake this cannot catch, which is why the
constant sits next to the rule with a comment saying so.

Like `fetch_hf.py` this is deliberately standalone -- it runs in the task's
environment image, which has `datasets`/`pandas`/`pyarrow` but not the
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
    # `fetch_hf.py` and the `autorlbench_datasets/` package are siblings in the
    # task's environment/ dir; put it on sys.path so the imports below resolve
    # regardless of the cwd prepare.sh runs us from.
    sys.path.insert(0, str(_HERE))

from fetch_hf import FetchError, materialize  # noqa: E402

# 派生数据集的缓存根。和 fetch_hf 的 DATASETS_ROOT 是同一个目录——
# 下载来的和现场转出来的都是数据集，没有理由分家。
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

    # "derived" 和 hf/ 并排：一个是下载来的，一个是照 recipe 转出来的，
    # 同一个 datasets 根下面分两棵树。
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
    args = parser.parse_args(argv)

    try:
        entry = build_cached(args.recipe, allow_count_mismatch=args.allow_count_mismatch)
        # No sha256/size check here: the recipe's own row-count and schema
        # assertions are the correctness gate for a derived dataset.
        materialize(entry, args.dest)
        # The marker is cache bookkeeping. Leaving it in the workspace would put
        # a stray dotfile in front of whatever globs the training script does.
        (args.dest / MARKER).unlink(missing_ok=True)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"[dataset] ready {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
