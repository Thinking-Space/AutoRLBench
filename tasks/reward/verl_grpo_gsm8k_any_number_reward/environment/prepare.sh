#!/usr/bin/env bash
# verl_grpo_gsm8k_any_number_reward: prepare stage.
#
# Runs in a separate prepare-stage container: root, the verl-0.9-vllm0.28
# image, **the only networked stage**, and $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are all writable.
#
# Two assets: the Qwen2.5-0.5B-Instruct checkpoint (988 MB) and the raw openai/gsm8k files (2.7 MB).
# The RL parquet files (data/gsm8k/{{train,test}}.parquet) ship with the workspace; they are
# what the unchanged verl 0.9.0 examples/data_preprocess/gsm8k.py produces from the raw files
# below, and run_gsm8k_grpo.sh rebuilds them from data/raw/gsm8k only if one is missing.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- image-level env, seeded here --------------------------------------------
# docker/images/verl-0.9-vllm0.28.Dockerfile is a build recipe baked without
# ENV; the two entries the recording rig's env carried ride the prepared/env
# handoff into the agent/rerun/verifier stages via the activate hook.
# WANDB_MODE/WANDB_API_KEY are exported by the harness itself for every
# stage. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions (and per-file pins) the former
# autorlbench_fetch_hf allowlist recorded, for reference:
#   Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --dest "$W/models/Qwen2.5-0.5B-Instruct"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/raw/gsm8k"
