#!/usr/bin/env bash
# verl_mbpp_sandbox_errors_pass: prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image,
# **the only networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
#
# run_grpo_mbpp.sh reads the model from $W/models/<name> and rebuilds data/<dataset>/*.parquet from the raw
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
# run_grpo_mbpp.sh rebuilds data/mbpp/{train,test}.parquet from these files with data_prep/mbpp.py.
# Pinned revision: google-research-datasets/mbpp @ 4bb6404fdc6cacfda99d4ac4205087b89d32030c
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo google-research-datasets/mbpp \
    --type dataset \
    --revision 4bb6404fdc6cacfda99d4ac4205087b89d32030c \
    --allow full/train-00000-of-00001.parquet \
    --allow full/validation-00000-of-00001.parquet \
    --allow full/prompt-00000-of-00001.parquet \
    --allow full/test-00000-of-00001.parquet \
    --expect full/train-00000-of-00001.parquet=09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae \
    --expect-size full/train-00000-of-00001.parquet=87223 \
    --expect full/validation-00000-of-00001.parquet=3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80 \
    --expect-size full/validation-00000-of-00001.parquet=25144 \
    --expect full/prompt-00000-of-00001.parquet=a053e4bb85ceb77430ae80592addb4ca4dc6ba087592f9e04537800ee88b7431 \
    --expect-size full/prompt-00000-of-00001.parquet=7878 \
    --expect full/test-00000-of-00001.parquet=566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77 \
    --expect-size full/test-00000-of-00001.parquet=115824 \
    --dest "$W/raw/mbpp"
