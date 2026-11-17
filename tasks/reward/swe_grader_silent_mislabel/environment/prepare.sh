#!/usr/bin/env bash
# Prepare stage: the only networked stage. It materialises the repository snapshots the corpus
# refers to and installs one pinned environment per repository from PyPI. Nothing here touches
# the grading pipeline or the corpus metadata.
set -euo pipefail
ENVIRONMENT_DIR="${AUTORLBENCH_ENVIRONMENT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
W="${AUTORLBENCH_WORKSPACE:?prepare requires the staged workspace}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

PY=$(command -v python3.10 || true)
if [ -z "$PY" ]; then
    PY=$(command -v python3)
fi
VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$VERSION" != "3.10" ]; then
    echo "prepare: the corpus environments are pinned to CPython 3.10, found $VERSION ($PY)" >&2
    exit 1
fi

python3 -B "$ENVIRONMENT_DIR/prepare_corpus.py" \
    --environment "$ENVIRONMENT_DIR" --workspace "$W" --python "$PY"
