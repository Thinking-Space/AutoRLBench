#!/usr/bin/env bash
set -euo pipefail
environment="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "$0")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:-$(cd "$environment/../workspace" && pwd)}"
prepared="${AUTORLBENCH_PREPARED:?the harness provides the prepared directory}"
(cd "$workspace" && sha256sum --strict --check overlay.sha256)
python3 "$environment/prepare_dependencies.py" "$prepared"
if [ ! -d "$prepared/accelerate-overlay/accelerate" ]; then
    mkdir -p "$prepared/accelerate-overlay"
    python3 "$environment/stage_accelerate_compat.py" --target "$prepared/accelerate-overlay"
fi
if [ ! -d "$prepared/k2v3/vllm" ]; then
    python3 "$workspace/reward_stock_k2v3_overlay.py" --overlay "$workspace/port" --target "$prepared/k2v3"
fi
# Nemotron snapshot 3441fcdf8c6e81a2959e6352ff50122e3c677d72.
python3 "$environment/prepare_fetch.py" --manifest "$environment/model_assets.json" --dest "$workspace/assets/nemotron-brorl"
(cd "$workspace" && sha256sum --strict --check data/input.sha256)
ln -f "$workspace/data/skywork_or1_7b_math_difficulty1-15_16k.parquet" "$workspace/assets/skywork_or1_7b_math_difficulty1-15_16k.parquet"
