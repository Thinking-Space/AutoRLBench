#!/usr/bin/env bash
# Prepare stage: the only networked stage. It unpacks the per-source query banks and rebuilds
# every query pool the case lists refer to, checking each rebuilt file against its recorded
# digest. No model, no dataset download, no GPU.
set -euo pipefail
ENVIRONMENT_DIR="${AUTORLBENCH_ENVIRONMENT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
W="${AUTORLBENCH_WORKSPACE:?prepare requires the staged workspace}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

python3 -B "$ENVIRONMENT_DIR/prepare_pools.py" --environment "$ENVIRONMENT_DIR" --workspace "$W"
