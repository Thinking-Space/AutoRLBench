#!/usr/bin/env bash
# Qwen3-1.7B / GSM-Hard — prepare stage.
#
# Runs in a separate prepare-stage container of the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`
# image; it is the only networked stage and $AUTORLBENCH_WORKSPACE is writable.
#
# run_tool_gsm_hard.sh reads the policy model from $W/models/Qwen3-1.7B and rebuilds
# data/gsm_hard/{train,test}.parquet from the raw Hugging Face snapshot under $W/raw/gsm-hard on every run
# (data_prep/gsm_hard.py, pandas only), so only these two pinned assets are fetched here.
# autorlbench_fetch.py is copied verbatim from infrastructure/verl_fsdp_microbatch_grad_reset and checks
# revision, byte size and sha256 of every pinned file.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Policy model -------------------------------------------------------------
# Pinned revision: Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
python3 "$env_dir/autorlbench_fetch.py" hf \
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
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect-size tokenizer.json=11422654 \
    --dest "$W/models/Qwen3-1.7B"

# --- Dataset ------------------------------------------------------------------
# Pinned revision: reasoning-machines/gsm-hard @ 960448f73503112d4226baeb8eb41d3fb5ae2506 (MIT)
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo reasoning-machines/gsm-hard \
    --type dataset \
    --revision 960448f73503112d4226baeb8eb41d3fb5ae2506 \
    --allow README.md \
    --allow gsmhardv2.jsonl \
    --expect gsmhardv2.jsonl=2032b0188a49ac0ec38cbe1e695e7d62d6dddc44bbb64f920728aca710e9ddef \
    --expect-size gsmhardv2.jsonl=1138151 \
    --dest "$W/raw/gsm-hard"
