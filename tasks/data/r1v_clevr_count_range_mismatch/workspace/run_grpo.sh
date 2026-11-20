#!/usr/bin/env bash
# GRPO on Qwen2-VL-2B-Instruct, CLEVR-CoGenT-A counting split (R1-V trainer).
# Trains, then evaluates every saved checkpoint on the two held-out sets.
set -euo pipefail
cd "$(dirname "$0")"
WS="$PWD"
export PYTHONPATH="$WS:${PYTHONPATH:-}"
export WANDB_MODE=offline

MODEL="${MODEL_DIR:-$WS/models/Qwen2-VL-2B-Instruct}"
DATA="${DATA_DIR:-$WS/data}"
LOG_DIR="${LOG_DIR:-$WS/results}"
OUT="${OUTPUT_DIR:-$WS/outputs/grpo}"
REWARDS="${REWARD_FUNCS:-count sformat}"; REWARDS="${REWARDS//,/ }"
MAX_STEPS="${MAX_STEPS:-400}"
LR="${LEARNING_RATE:-2e-6}"
BETA="${BETA:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SEED="${SEED:-42}"
mkdir -p "$LOG_DIR" "$OUT"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-12345}" \
    open_r1/grpo.py \
    --output_dir "$OUT" \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATA/train" \
    --deepspeed zero3.json \
    --reward_funcs $REWARDS \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --beta "$BETA" \
    --num_generations 8 \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_pixels 401408 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --logging_steps 1 \
    --save_steps "$SAVE_STEPS" \
    --save_only_model true \
    --seed "$SEED" \
    --report_to none 2>&1 | tee "$LOG_DIR/train.log"

python eval_checkpoints.py --base "$MODEL" --ckpt-root "$OUT" --data "$DATA" \
    --out "$LOG_DIR" --val-n "${VAL_N:-500}" --ood-n "${OOD_N:-1000}" 2>&1 | tee "$LOG_DIR/eval.log"
