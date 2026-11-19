#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"
E="$AUTORLBENCH_ENVIRONMENT"
W="$AUTORLBENCH_WORKSPACE"
# Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a.
# fetch_model.py calls the task-local, byte-identical fetch_hf.py and pins
# snapshot_download's revision. A revision-specific cache avoids floating HEAD.
# Destination is exactly the MODEL_DIR read by code/entry.sh.
python3 "$E/fetch_model.py"
python3 "$E/prepare_runtime.py"
# Vendored pure-Python mbridge @ 89eb10887887bc74853f89a4de258c0702932a1c.
# The Slime shared image already carries the compiled torch/CUDA/SG stack.
mkdir -p "$AUTORLBENCH_PREPARED/pysite"
cp -a "$W/dependencies/." "$AUTORLBENCH_PREPARED/pysite/"
printf 'PYTHONPATH=%s\n' "$AUTORLBENCH_PREPARED/pysite" > "$AUTORLBENCH_PREPARED/env"
# HF -> torch_dist conversion needs CUDA; code/entry.sh performs it on the
# first GPU launch. Prepare deliberately performs no GPU allocation/conversion.
