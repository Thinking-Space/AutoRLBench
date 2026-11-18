#!/usr/bin/env bash
# Qwen2.5 / GSM8K — prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image,
# **the only networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
#
# run_grpo_gsm8k.sh reads the model from $W/models/<name> and rebuilds data/<dataset>/*.parquet from the raw
# Hugging Face snapshot under $W/raw/<dataset> on every run (data_prep/*.py, pandas only), so only the
# raw files are fetched here. autorlbench_fetch.py (copied from configuration/verl_grpo_adamw_eps_too_large)
# checks revision, byte size and sha256 of every pinned file.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Policy model -------------------------------------------------------------
# Pinned revision: Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe \
    --expect-size model.safetensors=988097824 \
    --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
    --expect-size tokenizer.json=7031645 \
    --dest "$W/models/Qwen2.5-0.5B-Instruct"

# --- Dataset ------------------------------------------------------------------
# Pinned revision: openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow README.md \
    --allow main/train-00000-of-00001.parquet \
    --allow main/test-00000-of-00001.parquet \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$W/raw/gsm8k"
