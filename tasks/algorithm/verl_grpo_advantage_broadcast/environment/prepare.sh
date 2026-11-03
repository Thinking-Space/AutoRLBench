#!/usr/bin/env bash
# verl_grpo_advantage_broadcast — prepare stage.
#
# Runs in a separate prepare-stage container: root, `verl` image, **the only
# stage with network**. In the agent stage this file is mounted read-only at
# /autorlbench/environment.
#
# The upstream snapshot ships no per-file pins for these assets, so the
# task-local fetch_hf.py fetches the whole repos (cache-first, no checksum
# verification) instead of autorlbench_fetch_hf's pinned allowlist.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo cyh2004/ptdbench-taco-verl-easy \
    --dest "$AUTORLBENCH_WORKSPACE/taco_verl"
