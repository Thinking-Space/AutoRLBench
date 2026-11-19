#!/usr/bin/env bash
set -euo pipefail
environment="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "$0")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:-$(cd "$environment/../workspace" && pwd)}"
prepared="${AUTORLBENCH_PREPARED:?the harness provides the prepared directory}"
python3 "$environment/prepare_dependencies.py" "$prepared"
if [ ! -d "$prepared/accelerate-overlay/accelerate" ]; then
    mkdir -p "$prepared/accelerate-overlay"
    python3 "$environment/stage_accelerate_compat.py" --target "$prepared/accelerate-overlay"
fi
# Nemotron snapshot 3441fcdf8c6e81a2959e6352ff50122e3c677d72.
python3 "$environment/prepare_fetch.py" --manifest "$environment/model_assets.json" \
    --dest "$workspace/models/student"
(cd "$workspace" && sha256sum --strict --check data/input.sha256)
