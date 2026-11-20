#!/usr/bin/env python
"""Rebuild ``QAT/experiments/`` from the live run directories.

    $QAT_ROOT/tools/plotenv/bin/python scripts/build_experiment_archive.py

The archive is a *derived* artefact with one exception: the ``README.md`` inside
each day directory (and the top-level one) is written by hand and is preserved
across rebuilds.  Everything else -- FACTS.md, configs, curves, figures, the
code snapshot -- is deleted and regenerated on every run, so do not hand-edit
it; fix the harvester or the source files it reads instead.

Requires matplotlib (for the figures) and read access to the run directories;
both are checked up front and reported rather than worked around.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Names, not the module: ``exp_archive.harvest`` as an attribute resolves to the
# re-exported *function* of that name, not the submodule.
from exp_archive.harvest import RUNS_DIR, TRAIN_LOG_DIR, harvest  # noqa: E402
from exp_archive.render import render_archive  # noqa: E402


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="archive directory to (re)write; deleted first",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR, help="source runs/ directory"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=TRAIN_LOG_DIR, help="slurm log directory"
    )
    args = parser.parse_args()

    for path in (args.runs_dir, args.log_dir):
        if not path.is_dir():
            raise SystemExit(f"{path} is not a directory; set QAT_DATA_ROOT or pass --runs-dir")

    evals, trains = harvest(args.runs_dir, args.log_dir)
    if not evals and not trains:
        raise SystemExit(
            f"harvested nothing from {args.runs_dir}; refusing to write an empty archive "
            "that would read as 'no experiments were run'"
        )
    summary = render_archive(evals, trains, args.out, REPO_ROOT, _git_revision(REPO_ROOT))
    print(
        "wrote {out_dir}: {days} days, {trains} training jobs, {evals} evaluations, "
        "{figures} figures".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
