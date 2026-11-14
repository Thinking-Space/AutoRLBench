#!/usr/bin/env bash
# areal_kpop_binary_kl_fp32_nan_mask: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# `areal-v2.0.0-sglang0.5.10.post1-torch2.9.1-flashattn2.8.3-cu129-py3.12` image,
# **the only networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Base model ---------------------------------------------------------------
# run.sh sets MODEL_PATH="${MODEL_PATH:-$W/models/Qwen1.5-MoE-A2.7B-Chat}" and passes it as
# actor.path (the tokenizer, reference model and SGLang servers all resolve from it).
# Pinned revision, for reference: Qwen/Qwen1.5-MoE-A2.7B-Chat @ ec052fda178e241c7c443468d2fa1db6618996be
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --dest "$W/models/Qwen1.5-MoE-A2.7B-Chat"

# --- Dataset ------------------------------------------------------------------
# run.sh sets DATA_PATH="${DATA_PATH:-$W/data/gsm8k}" for both train_dataset.path and
# valid_dataset.path; areal/dataset/gsm8k.py loads it with load_dataset(path, name="main").
# Pinned revision, for reference: openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/gsm8k"
