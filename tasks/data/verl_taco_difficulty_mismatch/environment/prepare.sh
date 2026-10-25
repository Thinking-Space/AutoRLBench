#!/usr/bin/env bash
# task_train_too_hard: what to prepare before training.
#
# Migrated from environment/{Dockerfile,hydrate-assets.sh}. Runs in a separate prepare-stage
# container: root, `verl` environment activated, **the only stage with network**, and
# $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_PREPARED, $AUTORLBENCH_MODELS, $AUTORLBENCH_DATASETS are all writable.
#
# What the legacy Dockerfile did: download the model from HF, bake the dataset archive into the image,
# then have hydrate-assets.sh lay /opt/autorlbench-assets/workspace onto /workspace in the agent stage.
# Below, both steps move to the prepare stage with the revisions, byte sizes, and SHA-256 values
# recorded in the old Dockerfile kept verbatim, and the files land exactly where the legacy layout put them.
#
# In the agent stage this file is mounted read-only at /autorlbench/environment and the agent can read
# it: maintainer notes (migration background, relation to sibling tasks, grading thresholds) all go in
# ../README.md, not here.
set -euo pipefail

# --- Base model -------------------------------------------------------------
# In examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh:20,
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen2.5-1.5B-Instruct}"
# VERL_ROOT is two levels above the launch script, i.e. /workspace in the container, so dest must be
# exactly $AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct.
#
# The repo / revision / allow list / checksums all come verbatim from the legacy Dockerfile's
# MODEL_REPOSITORY, MODEL_REVISION, allow_patterns, MODEL_WEIGHT_SHA256, and the line
# `test "$(stat -c %s ...)" = 3087467144`. This snapshot repo is shared by the verl + TACO family of
# tasks and is not specific to this task.
autorlbench_fetch_hf \
    --repo LIF1014/autorlbench-verl-implementation-torch-functional \
    --revision 36a4c226ef70aef2c38df173a7cf86a47ad3ad00 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE \
    --allow merges.txt \
    --allow README.md \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow vocab.json \
    --allow model.safetensors \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expect-size model.safetensors=3087467144 \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"

# --- Dataset ----------------------------------------------------------------
# The same script (:21-22) requires $VERL_ROOT/taco_verl/{train,test}.parquet.
#
# There is no built-in recipe: upstream is a locally processed TACO snapshot whose filtering and split
# exist only in the published archive, with no recipe to recompute them. So this downloads a pinned
# archive instead of using autorlbench_dataset.
#
# The archive is a tar.gz with exactly two members: workspace/taco_verl/{train,test}.parquet
# (the legacy hydrate-assets.sh laid this workspace/ layer onto /workspace).
# autorlbench_fetch_hf does not unpack, so it downloads to a temp directory and is extracted here.
#
# Archive checksums come from DATASET_ARCHIVE_SHA256 / DATASET_ARCHIVE_SIZE in the legacy Dockerfile.
TACO_STAGING="$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"
rm -rf "$TACO_STAGING"

autorlbench_fetch_hf \
    --type dataset \
    --repo LIF1014/autorlbench-data-task-train-too-hard-dataset \
    --revision 22a1fee0ea2527f514c4115f6c792216a99d7e6f \
    --allow artifacts/workspace-assets.tar.gz \
    --expect artifacts/workspace-assets.tar.gz=eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916 \
    --expect-size artifacts/workspace-assets.tar.gz=167612568 \
    --dest "$TACO_STAGING"

echo "[prepare] unpacking taco_verl -> $AUTORLBENCH_WORKSPACE/taco_verl"
tar -xzf "$TACO_STAGING/artifacts/workspace-assets.tar.gz" \
    -C "$TACO_STAGING" workspace/taco_verl

# mv instead of cp: on the same filesystem it is a single rename, avoiding a second 220MB copy.
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
mv "$TACO_STAGING/workspace/taco_verl" "$AUTORLBENCH_WORKSPACE/taco_verl"
rm -rf "$TACO_STAGING"

# Re-check the extracted bytes. autorlbench_fetch_hf checks the archive; this checks the two files
# training actually reads. A tar step sits between them, so each is worth checking separately.
# On mismatch, fail hard and do not re-download. Checksums come from the data repo's published revision.
(
    cd "$AUTORLBENCH_WORKSPACE/taco_verl"
    sha256sum -c - <<'TACO_DIGESTS'
0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3  test.parquet
8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573  train.parquet
TACO_DIGESTS
    [ "$(stat -c %s test.parquet)" = 61082426 ]
    [ "$(stat -c %s train.parquet)" = 168834942 ]
)

# pyext is not installed: verl/utils/reward_score/taco.py first tries `from pyext import RuntimeModule`
# and on ImportError falls back to its bundled _FallbackRuntimeModule (fully functional, not partial).
# The frozen dependency list of the recorded run has no pyext; installing it would change the reward.
