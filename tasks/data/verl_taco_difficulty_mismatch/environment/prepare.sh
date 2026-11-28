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
    --repo LIF1014/ptdbench-data-task-train-too-hard-dataset \
    --cache-as autorlbench/taco-asset \
    --dest "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"

# The dataset repo lives under its original public name: the 2026-10-19
# PTDBench -> AutoRLBench rename of references in this repository never
# happened on the Hub (the autorlbench-* id 401s for everyone without a warm
# cache), so fetch the id the pinned revision actually lives at. The repo id
# still states this task's answer -- see the Known leak note in ../README.md.
#
# The archive's immutable bytes, per ../README.md (revision 22a1fee0…,
# 167,612,568 bytes, SHA-256 eddeece8…). fetch_hf.py verifies nothing and
# follows HEAD, so pin the archive here: a moved revision or a rewritten
# archive fails loudly instead of silently changing the training data. This
# restores the checksum guard the 2026-10-19 refactor dropped together with
# the legacy ptdbench_fetch_hf's --revision/--expect flags.
TACO_ARCHIVE="$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive/artifacts/workspace-assets.tar.gz"
test "$(wc -c < "$TACO_ARCHIVE")" -eq 167612568
printf '%s  %s\n' \
  eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916 "$TACO_ARCHIVE" | sha256sum -c -

echo "[prepare] unpacking taco_verl -> $AUTORLBENCH_WORKSPACE/taco_verl"
tar -xzf "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive/artifacts/workspace-assets.tar.gz" \
    -C "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive" workspace/taco_verl

# mv instead of cp: on the same filesystem it is a single rename, avoiding a
# second 220MB copy.
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
mv "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive/workspace/taco_verl" "$AUTORLBENCH_WORKSPACE/taco_verl"
rm -rf "$AUTORLBENCH_WORKSPACE/.autorlbench-taco-archive"
