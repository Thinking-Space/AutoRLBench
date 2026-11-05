#!/usr/bin/env bash
# Prepare stage, shared byte-identical by the verl-v061-grpo task pair
# (verl_policy_loss_padded_mask_only, verl_policy_loss_padded_unmasked).
#
# Runs in the prepare-stage container: root, the verl-v061-grpo image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It fetches the
# Qwen3-1.7B weights, the DAPO-Math-17k parquet and the AIME24/25 sets, then
# derives the training/eval assets via prepare_data.py, which verifies every
# file against asset_registry.json (the per-file pins live there).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?The official harness sets this}"
workspace="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$workspace/assets/raw"

# --- image-level env, seeded here --------------------------------------------
# docker/images/verl-v061-grpo.Dockerfile is a build recipe baked without
# ENV; the two entries the recording rig's image carried ride the
# prepared/env handoff into the agent/rerun/verifier stages via the activate
# hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification (asset_registry.json is the pin of record); the
# revisions the former autorlbench_fetch_hf allowlist pinned, for reference:
#   Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
#   BytedTsinghua-SIA/DAPO-Math-17k @ 61829aa9fe754403c5559e46423772deac5a329b
#   HuggingFaceH4/aime_2024 @ 2fe88a2f1091d5048c0f36abc874fb997b3dd99a
#   yentinglin/aime_2025 @ 6f71d77b0b89b9dabe07ab466c51df33f514df7f
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen3-1.7B \
    --dest "$workspace/model"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo BytedTsinghua-SIA/DAPO-Math-17k \
    --dest "$workspace/assets/raw/dapo"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo HuggingFaceH4/aime_2024 \
    --dest "$workspace/assets/raw/aime24"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo yentinglin/aime_2025 \
    --dest "$workspace/assets/raw/aime25"
python "$env_dir/prepare_data.py" --workspace "$workspace" --registry "$env_dir/asset_registry.json"
