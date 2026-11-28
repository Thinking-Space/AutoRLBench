#!/usr/bin/env bash
# slime_moe_router_bias_bf16_corruption: no external assets are required.
# Runs in the shared Slime v0.3.2 image; the public reproduction is self-contained.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_PREPARED:?}"
mkdir -p "$AUTORLBENCH_PREPARED"
: > "$AUTORLBENCH_PREPARED/env"
test -s "$AUTORLBENCH_WORKSPACE/train.py"
test -s "$AUTORLBENCH_WORKSPACE/checkpoint.py"
