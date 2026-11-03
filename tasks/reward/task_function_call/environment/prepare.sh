#!/usr/bin/env bash
# task_function_call — prepare stage.
#
# Runs in a separate prepare-stage container: root, the task's declared image,
# **the only stage with network**. In the agent stage this file is mounted
# read-only at /autorlbench/environment. $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_PREPARED, $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS are all
# writable.
#
# This script only places the model and dataset where the launch script reads
# them. Why exactly these two repos, why pyext is not installed, and what the
# migration filled in are all recorded in README.md; maintainers read that and
# must not write answer-like background back here — this file is visible to the
# agent during evaluation.
#
# Recorded assets are fetched whole-repo with the task-local fetch_hf.py
# (cache-first under $AUTORLBENCH_MODELS, no revision/sha pins — converting to
# pinned autorlbench_fetch_hf would change fetch behavior).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# --- base model --------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:20 sets
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# where VERL_ROOT is two levels up from the launch script — /workspace inside
# the container — so dest must be exactly $AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct.
# solution/instruction.md also pins MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct in
# "The effective run must use", and the historical log at line :83 records the
# same vLLM model path — all three places must agree.
#
# Official repo, fetched whole (no --allow subset, no sha256 check in fetch).
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- dataset -----------------------------------------------------------------
# The same launch script at :21-22 reads $VERL_ROOT/taco_verl/{train,test}.parquet;
# the historical log at :1599/:1605 likewise records /workspace/taco_verl/{train,test}.parquet.
#
# No built-in recipe: the upstream is a local TACO EASY snapshot (picture
# problems with picture_num != 0 dropped), 8368 train / 184 test; that filter
# and split exist only in the published snapshot, with no recomputable recipe.
# So the flow is "fetch a pinned archive" rather than an HF dataset pipeline.
#
# The archive is a tar.gz containing workspace/taco_verl/*.parquet. fetch_hf
# does not unpack, so download into a staging directory and expand manually;
# then verify each of the two parquet files.
#
# Digest values come from artifacts/MANIFEST.json in the repo and match the two
# rows of the README.md "External dataset asset" table.
TACO_STAGING="$AUTORLBENCH_WORKSPACE/.taco-archive"
rm -rf "$TACO_STAGING"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo LIF1014/ptdbench-verl-coding-tasks-function-call-dataset \
    --cache-as verl-asset-2 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $AUTORLBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv rather than cp: on the same filesystem this is a rename, saving a second
# 667MB copy.
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$AUTORLBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# Verify the two expanded parquet files independently (fetch itself no longer
# checks); training reads them directly. A mismatch is a hard failure.
(
    cd "$AUTORLBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa  test.parquet
7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082676 ]
    [ "$(stat -c %s train.parquet)" = 606016648 ]
)
