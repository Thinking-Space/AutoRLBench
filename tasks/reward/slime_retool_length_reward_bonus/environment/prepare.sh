#!/usr/bin/env bash
# slime_retool_length_hack: what to prepare before training.
#
# Identical to tasks/slime_retool/environment/prepare.sh, because both tasks run the
# **same recipe** (examples/retool/retool_qwen3_4b_rl.sh) and differ only in the length bonus
# inside reward_func. The healthy/unhealthy pair must differ only by the injected defect: if the
# data, weights, or ref checkpoint differ anywhere, comparing the two curves no longer shows anything.
# Before changing this file, check whether the other copy needs the same change.
#
# Runs in a separate prepare-stage container: root, `slime-v0.3.2` image, **the only stage with
# network**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are all writable.
#
# Four assets: three downloaded, one converted on the spot. Directory names are set by the
# DATA_DIR/MODEL_DIR guards in examples/retool/retool_qwen3_4b_rl.sh (both default to $PWD, which is
# /workspace in the container), and **must not change by a single character**: otherwise the guard
# misses, the script silently passes a nonexistent path to --hf-checkpoint, and the error surfaces in
# megatron, far from the cause.
set -euo pipefail

# --- Training set -----------------------------------------------------------
# --prompt-data + --input-key prompt + --label-key label read a jsonl.
# Note this is **not the same** as DAPO-Math-17k_verl in the verl tasks: that one uses a parquet
# schema. Do not merge them to "unify" things; they serve two loaders in two frameworks.
autorlbench_fetch_hf \
    --repo zhuzilin/dapo-math-17k \
    --type dataset \
    --revision 2e65612930298bde4c5d58fd97b3f23a483aaff9 \
    --allow dapo-math-17k.jsonl \
    --expect dapo-math-17k.jsonl=cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6 \
    --expect-size dapo-math-17k.jsonl=10490834 \
    --dest "$AUTORLBENCH_WORKSPACE/zhuzilin_dapo-math-17k"

# --- Validation set ---------------------------------------------------------
# Source of `--eval-prompt-data aime "$EVAL_DATA"` in EVAL_ARGS, i.e. the primary metric eval/aime.
# 30 problems, and this is the curve the grading script looks at. This task's outcome records exactly
# that the length bonus pushes eval/aime up to +3.84 while held-out accuracy drops.
autorlbench_fetch_hf \
    --repo zhuzilin/aime-2024 \
    --type dataset \
    --revision 1c625e328db94ec7ef7ff169016b097c468d60b9 \
    --allow aime-2024.jsonl \
    --expect aime-2024.jsonl=5d4685dd504391b0ec00a9b73c379a80ba4da804829a113b1d4fe630308d47cc \
    --expect-size aime-2024.jsonl=13029 \
    --dest "$AUTORLBENCH_WORKSPACE/zhuzilin_aime-2024"

# --- SFT'd HF checkpoint (--hf-checkpoint) ----------------------------------
# The allow list excludes .gitattributes; everything else is tokenizer / config the model reads itself.
# Only the two shards get a sha256: the revision already pins the small files, and listing each one
# would just make the task package harder to write.
autorlbench_fetch_hf \
    --repo font-info/qwen3-4b-sft-SGLang-RL \
    --revision 6de9b0b89be3deddd78e9245670164ed9145ded4 \
    --allow LICENSE \
    --allow added_tokens.json \
    --allow chat_template.jinja \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00002.safetensors \
    --allow model-00002-of-00002.safetensors \
    --allow model.safetensors.index.json \
    --allow special_tokens_map.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00002.safetensors=9164ce24732079f2515a3c8c29d6537abf8b56d2a64915f1572b6fa6ab4719ee \
    --expect-size model-00001-of-00002.safetensors=4862372272 \
    --expect model-00002-of-00002.safetensors=35dac1e2a58d6f1cd9d193ce11070d29536d6822eaf0f44dba9eb39a3c0fe405 \
    --expect-size model-00002-of-00002.safetensors=3960522232 \
    --dest "$AUTORLBENCH_WORKSPACE/font-info_qwen3-4b-sft"

# --- ref weights in mcore (torch_dist) format (--ref-load) ------------------
# There is no upstream repo; they can only be converted on the spot from the checkpoint above. The
# original recipe converts inside the training script (retool_qwen3_4b_rl.sh:68-76); doing it here
# ahead of time saves GPU hours for the training itself.
#
# Two pitfalls:
#
#   * The output is **not guaranteed byte-reproducible**: the dist-checkpoint shard layout follows the
#     world size. Pinning WORLD_SIZE to 1 yields a single pp=1 layout whose shape does not change
#     across machines or GPU counts (convert_hf_to_torch_dist.py:62-79 derives pp size from WORLD_SIZE).
#     The 4B bf16 weights are about 8GB and fit on one GPU.
#   * --rotary-base must be 5000000, not the default 1000000 from scripts/models/qwen3-4B.sh:13. The
#     RL script passes it again **after** MODEL_ARGS (:74 and :84), and the later one wins; conversion
#     and training must use the same value for the ref weights' positional encoding to match the actor.
#     It is hardcoded rather than inherited from the environment, so MODEL_ARGS_ROTARY_BASE cannot
#     silently change it some day.
#
# The slime source tree lives in the **task workspace**, not in the image; this task ships its own.
SLIME_DIR="$AUTORLBENCH_WORKSPACE"
HF_MODEL_DIR="$AUTORLBENCH_WORKSPACE/font-info_qwen3-4b-sft"
DIST_MODEL_DIR="$AUTORLBENCH_WORKSPACE/font-info_qwen3-4b-sft_torch_dist"

# Conversion needs CUDA + NCCL (dist.init_process_group + torch.cuda.set_device).
# The prepare stage gets GPUs via NVIDIA_VISIBLE_DEVICES from metadata, so on the normal path this
# holds; but the harbor backend runs prepare as a build-time RUN layer, which has neither the task env
# nor a GPU. So this **skips instead of failing**: the training script's own
# `if [ ! -d "$DIST_MODEL_DIR" ]` guard redoes it in the agent stage, when /workspace is still
# writable. Skipping it here is just slower; failing to do it at all would be the real error.
if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1, rotary-base=5000000)"
    # Source MODEL_ARGS from the task's own copy instead of duplicating it here; a copy would drift from the recipe.
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen3-4B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="/root/Megatron-LM:$SLIME_DIR" \
    python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --rotary-base 5000000 \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to the"
    echo "[prepare] training script's own guard (examples/retool/retool_qwen3_4b_rl.sh)."
fi
