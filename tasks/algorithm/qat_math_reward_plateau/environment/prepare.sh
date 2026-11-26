#!/usr/bin/env bash
set -euo pipefail

environment=$(cd "$(dirname "$0")" && pwd)
workspace=${AUTORLBENCH_WORKSPACE:?the harness provides the workspace}
prepared=${AUTORLBENCH_PREPARED:?the harness provides the prepared directory}
mkdir -p "$workspace/assets" "$prepared/accelerate-overlay"

cd "$workspace"
sha256sum --strict --check recorded.sha256 data/input.sha256 overlay.sha256
python3 "$environment/prepare_dependencies.py" "$prepared"
python3 "$environment/stage_accelerate_compat.py" --target "$prepared/accelerate-overlay"
python3 "$workspace/reward_stock_k2v3_overlay.py" --overlay "$workspace/port" --target "$prepared/k2v3"
python3 "$environment/prepare_fetch.py" --manifest "$environment/model_assets.json" \
    --dest "$workspace/assets/nemotron-brorl"
ln -f "$workspace/data/skywork_or1_7b_math_difficulty1-15_16k.parquet" \
    "$workspace/assets/skywork_or1_7b_math_difficulty1-15_16k.parquet"
