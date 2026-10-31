#!/usr/bin/env bash
# verl_agent_loop_template_retokenization: what to prepare before training.
#
# Runs in a separate prepare-stage container: root, the `verl-0.9-vllm0.28` image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are
# all writable.
#
# Three assets: the Qwen3-1.7B checkpoint (4.1 GB), the raw openai/gsm8k parquet files (2.7 MB) and
# the raw GSM-Hard jsonl (1.1 MB). The RL parquet files (data/gsm8k/{train,test,test_hard}.parquet)
# ship with the workspace; they are what calc_agent/prepare_gsm8k.py produces from the raw files
# below, and run_calc_grpo.sh rebuilds them from data/raw/ only if one is missing.
set -euo pipefail

# --- Base model ---------------------------------------------------------------
# run_calc_grpo.sh reads MODEL_PATH=$W/models/Qwen3-1.7B and refuses to start when
# $MODEL_PATH/config.json is missing. Revision and shard sha256s are the ones the recorded runs
# loaded (the HF etags of the local snapshot).
autorlbench_fetch_hf \
    --repo Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00002.safetensors \
    --allow model-00002-of-00002.safetensors \
    --allow model.safetensors.index.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00002.safetensors=169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 \
    --expect-size model-00001-of-00002.safetensors=3441185608 \
    --expect model-00002-of-00002.safetensors=912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 \
    --expect-size model-00002-of-00002.safetensors=622329984 \
    --dest "$AUTORLBENCH_WORKSPACE/models/Qwen3-1.7B"

# --- Dataset ------------------------------------------------------------------
# run_calc_grpo.sh checks $W/data/raw/gsm8k/main/train-00000-of-00001.parquet before starting.
autorlbench_fetch_hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow main/train-00000-of-00001.parquet \
    --allow main/test-00000-of-00001.parquet \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$AUTORLBENCH_WORKSPACE/data/raw/gsm8k"

# run_calc_grpo.sh checks $W/data/raw/gsm-hard/gsmhardv2.jsonl before starting (second held-out set).
autorlbench_fetch_hf \
    --repo reasoning-machines/gsm-hard \
    --type dataset \
    --revision 960448f73503112d4226baeb8eb41d3fb5ae2506 \
    --allow gsmhardv2.jsonl \
    --expect gsmhardv2.jsonl=2032b0188a49ac0ec38cbe1e695e7d62d6dddc44bbb64f920728aca710e9ddef \
    --expect-size gsmhardv2.jsonl=1138151 \
    --dest "$AUTORLBENCH_WORKSPACE/data/raw/gsm-hard"
