#!/usr/bin/env bash
# verl_taco_difficulty_mismatch — prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image, **the only
# stage with network**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_PREPARED,
# $AUTORLBENCH_MODELS, $AUTORLBENCH_DATASETS are all writable.
#
# In the agent stage this file is mounted read-only at /autorlbench/environment
# and the agent can read it: maintainer notes (migration background, relation to
# sibling tasks, grading thresholds) all go in ../README.md, not here.
#
# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the pinned revisions, byte sizes, and SHA-256 values
# are recorded in ../README.md for reference.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# --- Base model -------------------------------------------------------------
# In examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:22,
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# VERL_ROOT is two levels above the launch script, i.e. the workspace root in
# the container, so dest must be exactly $AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct.
#
# The official Qwen repo, byte-identical on every allowed file to the former
# family snapshot repo (LIF1014/autorlbench-verl-implementation-torch-functional@36a4c22,
# same model.safetensors sha256). The switch matters: that snapshot repo's name
# is visible to the agent through the shared download cache and names a sibling
# task's defect; the official repo name leaks nothing.
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- Dataset ----------------------------------------------------------------
# The same script (:23-24) requires $VERL_ROOT/taco_verl/{train,test}.parquet.
#
# There is no built-in recipe: upstream is a locally processed TACO snapshot
# whose filtering and split exist only in the published archive, with no recipe
# to recompute them. So this downloads the pinned archive instead of using a
# dataset recipe.
#
# The archive is a tar.gz with exactly two members: workspace/taco_verl/{train,test}.parquet.
# fetch_hf.py does not unpack, so it downloads to a staging directory and is
# extracted here.
#
# The repo name states this task's answer; --cache-as keeps it out of the
# shared download cache, which is mounted read-only into the agent sandbox on
# some backends. The name is agent-readable inside this script either way —
# see the Known leak note in ../README.md; republishing the archive under a
# neutral repository name is still open.
rm -rf "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo LIF1014/autorlbench-data-task-train-too-hard-dataset \
    --cache-as autorlbench/taco-asset \
    --dest "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"

echo "[prepare] unpacking taco_verl -> $AUTORLBENCH_WORKSPACE/taco_verl"
tar -xzf "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive/artifacts/workspace-assets.tar.gz" \
    -C "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive" workspace/taco_verl

# mv instead of cp: on the same filesystem it is a single rename, avoiding a
# second 220MB copy.
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
mv "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive/workspace/taco_verl" "$AUTORLBENCH_WORKSPACE/taco_verl"
rm -rf "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"
