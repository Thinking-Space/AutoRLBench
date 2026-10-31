#!/usr/bin/env bash
# verl_grpo_ppo_mini_batch_too_small: what to prepare before training.
#
# Runs in a separate prepare-stage container: root, the `verl-0.9-vllm0.28` image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are
# all writable.
#
# Two assets: the Qwen2.5-0.5B-Instruct checkpoint (988 MB) and the raw openai/gsm8k files (2.7 MB).
# The RL parquet files (data/gsm8k_boxed/{train,test}.parquet) ship with the workspace; they are what the
# shipped examples/data_preprocess/gsm8k_boxed.py (upstream gsm8k.py with the \boxed{} instruction) produces from the raw files below, and
# run_gsm8k_grpo.sh rebuilds them from data/raw/gsm8k only if one is missing.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# run_gsm8k_grpo.sh reads MODEL_PATH=$W/models/Qwen2.5-0.5B-Instruct and refuses to start when
# $MODEL_PATH/config.json is missing. Revision and sha256 are those of the snapshot the recorded
# runs loaded.
autorlbench_fetch_hf \
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
    --dest "$AUTORLBENCH_WORKSPACE/models/Qwen2.5-0.5B-Instruct"

# --- Dataset ------------------------------------------------------------------
# examples/data_preprocess/gsm8k.py calls datasets.load_dataset($W/data/raw/gsm8k, "main"), which needs
# the repo README.md (config metadata) next to main/{train,test}-00000-of-00001.parquet.
autorlbench_fetch_hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow README.md \
    --allow main/train-00000-of-00001.parquet \
    --allow main/test-00000-of-00001.parquet \
    --expect README.md=a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1 \
    --expect-size README.md=7935 \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$AUTORLBENCH_WORKSPACE/data/raw/gsm8k"
