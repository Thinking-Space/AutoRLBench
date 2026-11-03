#!/usr/bin/env bash
# reward_parser_mismatch — prepare stage.
#
# Runs in a separate prepare-stage container: root, the task's declared image,
# **the only stage with network**. In the agent stage this file is mounted
# read-only at /autorlbench/environment. $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_PREPARED, $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS are all
# writable.
#
# Recorded assets are fetched whole-repo with the task-local fetch_hf.py
# (cache-first under $AUTORLBENCH_MODELS, no revision/sha pins — converting to
# pinned autorlbench_fetch_hf would change fetch behavior).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# --- base model --------------------------------------------------------------
# run_buggy.sh resolves the model from the MODEL_PATH env var (the rerun stage
# sets MODEL_PATH=/workspace/Qwen3-4B-Base), so dest must be exactly
# $AUTORLBENCH_WORKSPACE/Qwen3-4B-Base.
#
# Official repo, fetched whole (no --allow subset, no sha256 check in fetch).
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo "Qwen/Qwen3-4B-Base" \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen3-4B-Base"
