#!/usr/bin/env bash
# What to prepare before training.
#
# Runs in a separate prepare-stage container: root, the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image, **the only networked
# stage**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are all
# writable.
#
# Two assets: the Qwen2.5-0.5B-Instruct checkpoint (988 MB) and the raw openai/gsm8k files (2.7 MB),
# at the same revisions and sha256 as tasks/reward/verl_grpo_gsm8k_any_number_reward.
# The RL parquet files (data/gsm8k/{train,test}.parquet) ship with the workspace, byte-identical
# to what the unchanged verl examples/data_preprocess/gsm8k.py produces from the raw files below
# (verified on the eval image: sha256s match the grader's pins); run_gsm8k_grpo.sh keeps its
# rebuild-from-data/raw branch as a fallback, but it must not run in this harness — the image has
# no pip-installed verl and gsm8k.py imports verl before run_gsm8k_grpo.sh exports PYTHONPATH.
#
# Host memory: the recorded runs peaked at 25 GB RSS on one GPU with the worker counts set in
# run_gsm8k_grpo.sh (dataloader 0, agent loop 1, reward loop 1); verl's defaults (8/8/8) needed
# more than 28 GB.
#
# The fetch script ships with the task (environment/autorlbench_fetch.py, the pinned fetcher that was
# docker/autorlbench_fetch.py before 2026-11-05): it checks revision, size and sha256, which the solution
# grader relies on for the raw GSM8K files.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# run_gsm8k_grpo.sh reads MODEL_PATH=$W/models/Qwen2.5-0.5B-Instruct and refuses to start when
# $MODEL_PATH/config.json is missing. Revision and sha256 are those of the snapshot the recorded
# runs loaded.
python3 "$AUTORLBENCH_ENVIRONMENT/autorlbench_fetch.py" hf \
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
python3 "$AUTORLBENCH_ENVIRONMENT/autorlbench_fetch.py" hf \
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
