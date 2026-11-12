#!/usr/bin/env bash
# torch_functional — prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image, **the only
# stage with network**. In the agent stage this file is mounted read-only at
# /autorlbench/environment. $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_PREPARED,
# $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS are all writable.
#
# Migrated from the legacy environment/{Dockerfile,hydrate-assets.sh} pair:
# the old flow baked a patched pyext plus model/dataset archives into the
# image, then a hydrate step laid the archived workspace over /workspace in
# the agent stage. Here, "fetch model + fetch dataset" is
# restored into the prepare stage verbatim (revisions, byte sizes and
# SHA-256 values as recorded in README.md); the pyext section is
# deliberately NOT installed — rationale at the end.
#
# Recorded assets are fetched whole-repo with the task-local fetch_hf.py
# (cache-first under $AUTORLBENCH_MODELS, no revision/sha pins — converting to
# pinned autorlbench_fetch_hf would change fetch behavior).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# --- base model -------------------------------------------------------------
# examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:20 has
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# where VERL_ROOT is two levels up from the launch script — inside the
# container that is /workspace, so dest must be exactly
# $AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct.
#
# Official repo, fetched whole (no --allow subset, no sha256 check in fetch).
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- dataset ----------------------------------------------------------------
# The same launch script :21-22 reads $VERL_ROOT/taco_verl/{train,test}.parquet.
#
# No built-in recipe: the upstream is a local TACO EASY selection (drop items
# with picture_num != 0), 8368 train / 184 test — that filter and split exist
# only in the published snapshot, there is no recomputable recipe. So we fetch
# a pinned archive instead of using a dataset recipe. The two parquet files
# are byte-identical to task_function_call's (README.md "External dataset
# asset" table), but the archive is published in this task's own dataset
# repo, whose archive SHA-256 / sizes differ from task_function_call's.
#
# The archive is a tar.gz containing workspace/taco_verl/*.parquet (the legacy
# hydrate-assets.sh laid exactly that workspace/ layer over /workspace).
# fetch_hf.py does not unpack, so download into a staging dir and expand here;
# then verify each parquet individually.
#
TACO_STAGING="$AUTORLBENCH_WORKSPACE/.taco-archive"
rm -rf "$TACO_STAGING"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo LIF1014/ptdbench-verl-implementation-torch-functional-dataset \
    --cache-as verl-asset-1 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $AUTORLBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv rather than cp: on the same filesystem it is a rename, saving a second
# 667MB copy.
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$AUTORLBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# Verify the two unpacked parquet files ourselves (fetch no longer checks):
# training actually reads these. Any mismatch is a hard failure.
# Digests and sizes taken from the README.md "External dataset asset" table.
(
    cd "$AUTORLBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa  test.parquet
7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082676 ]
    [ "$(stat -c %s train.parquet)" = 606016648 ]
)

# --- pyext: deliberately NOT installed here ---------------------------------
# verl/utils/reward_score/taco.py first tries `from pyext import RuntimeModule`;
# on ImportError it falls back to its own _FallbackRuntimeModule (complete,
# not a stub), and the stdin path runs for every sample — so whether pyext is
# installed lands in the reward.
#
# Not installing it is justified by this task's frozen wandb manifest:
#   workspace/logs/wandb/*/files/requirements.txt has no pyext.
# The benchmark thresholds are calibrated on the historical curves (README.md:
# val peak >= 0.125, critic/rewards/mean > 0.14), so the wandb manifest is
# authoritative, not the legacy Dockerfile. (task_function_call makes the same
# call; task_evaluator installs it because its requirements.txt has
# pyext==0.7.)
#
# To reverse this: copy the section at the end of
# reward_design/task_evaluator/environment/prepare.sh (installs into
# $AUTORLBENCH_PREPARED/pysite and writes PYTHONPATH into
# $AUTORLBENCH_PREPARED/env).
# Do not modify docker/images/verl.Dockerfile — that image is shared by every
# verl task.
