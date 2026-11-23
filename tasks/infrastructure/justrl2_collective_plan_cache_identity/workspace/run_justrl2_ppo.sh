#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HF_MODEL_DIR=${HF_MODEL_DIR:-$PWD/models/JustRL-II-base-model}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-$PWD/models/JustRL-II-base-model-torch_dist}
RAW_DATA=${RAW_DATA:-$PWD/raw/countdown}
DATA_DIR=${DATA_DIR:-$PWD/data/countdown}
OUT=${OUT:-${LOG_DIR:-$PWD/results/ppo_countdown}}
STEPS=${STEPS:-30}
SEED=${SEED:-42}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS:-3}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
LR=${LR:-2e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-3}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

mkdir -p "$OUT" "$DATA_DIR"

if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/test.jsonl" ]; then
  if [ -e "$RAW_DATA" ]; then
    python3 data_prep/countdown.py --src "$RAW_DATA" --out "$DATA_DIR" 2>&1 | tee "$OUT/data_prep.log"
  else
    echo "FATAL: missing prepared $DATA_DIR/{train,test}.jsonl and RAW_DATA=$RAW_DATA" >&2
    exit 1
  fi
fi

# Convert HF model -> Megatron torch_dist on first GPU run if prepare.sh ran CPU-only
if [ ! -d "$MEGATRON_MODEL_PATH/release" ] && [ ! -f "$MEGATRON_MODEL_PATH/latest_checkpointed_iteration.txt" ]; then
  echo "Converting HF model $HF_MODEL_DIR -> $MEGATRON_MODEL_PATH ..."
  mkdir -p "$MEGATRON_MODEL_PATH"
  source justrl2/model_args/minicpm5-2b.sh
  HF_TRUST_REMOTE_CODE=1 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  NVTE_FUSED_ATTN=1 \
  NVTE_FLASH_ATTN=0 \
  NVTE_UNFUSED_ATTN=0 \
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${CONVERT_MASTER_PORT:-29365}" \
  PYTHONPATH="$PWD:${MEGATRON_DIR}:${SGLANG_DIR}/python${PYTHONPATH:+:$PYTHONPATH}" \
    python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --attention-backend fused \
      --no-gradient-accumulation-fusion \
      --hf-checkpoint "$HF_MODEL_DIR" \
      --save "$MEGATRON_MODEL_PATH" 2>&1 | tee "$OUT/convert_model.log"
fi

rm -f "$OUT/metrics.jsonl"
rm -rf "$OUT/val_generations"
mkdir -p "$OUT/val_generations"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY ROCR_VISIBLE_DEVICES

export PYTHONPATH="$PWD:${MEGATRON_DIR}:${SGLANG_DIR}/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_TRUST_REMOTE_CODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0
export NVTE_UNFUSED_ATTN=0
export NCCL_DEBUG=WARN
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$OUT/nccl_trace_rank_"
export WANDB_MODE=offline
export TENSORBOARD_DIR="$OUT/tensorboard"
export RAY_DEDUP_LOGS=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0
export MILES_METRICS_FILE="$OUT/metrics.jsonl"
export MILES_EVAL_DUMP_DIR="$OUT/val_generations"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$$}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TENSORBOARD_DIR" "$TRITON_CACHE_DIR"

source justrl2/model_args/minicpm5-2b.sh

RAY_TMPDIR=$(mktemp -d /tmp/justrl2-ray-XXXXXX)
cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR" "$TRITON_CACHE_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

RAY_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); p=s.getsockname()[1]; s.close(); print(p)')
ray start --head \
  --node-ip-address=127.0.0.1 \
  --port="$RAY_PORT" \
  --temp-dir="$RAY_TMPDIR" \
  --num-gpus=8 \
  --num-cpus=32 \
  --object-store-memory=8000000000 \
  --include-dashboard=false \
  --disable-usage-stats

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

TRAIN_ENV_JSON=$(python3 -c '
import json, os
keys = [
    "PYTHONPATH", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "CUDA_DEVICE_MAX_CONNECTIONS", "NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN",
    "NCCL_DEBUG", "NCCL_CUMEM_ENABLE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    "WANDB_MODE", "TENSORBOARD_DIR",
    "MILES_METRICS_FILE", "MILES_EVAL_DUMP_DIR",
    "TRITON_CACHE_DIR", "TMPDIR",
]
print(json.dumps({k: os.environ[k] for k in keys if k in os.environ}))
')

python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 4 \
  --num-gpus-per-node 8 \
  --colocate \
  --no-offload-train \
  --no-offload-rollout \
  --train-backend megatron \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --qkv-format thd \
  --attention-backend fused \
  --no-gradient-accumulation-fusion \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --log-probs-chunk-size -1 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_MODEL_DIR" \
  --ref-load "$MEGATRON_MODEL_PATH" \
  --prompt-data "$DATA_DIR/train.jsonl" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --rollout-shuffle \
  --custom-rm-path reward.countdown_reward.compute_reward \
  --num-rollout "$STEPS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --rollout-max-prompt-len 1024 \
  --rollout-max-response-len 512 \
  --rollout-max-context-len 2048 \
  --seq-length 2048 \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --rollout-stop-token-ids 1 130073 \
  --rollout-seed "$SEED" \
  --seed "$SEED" \
  --advantage-estimator ppo \
  --gamma 1.0 \
  --lambd 1.0 \
  --gae-lambda-k 0.5 \
  --critic-value-bias-init 0.25 \
  --num-critic-only-steps "$NUM_CRITIC_ONLY_STEPS" \
  --critic-lr "$CRITIC_LR" \
  --critic-lr-warmup-iters "$NUM_CRITIC_ONLY_STEPS" \
  --normalize-advantages \
  --calculate-per-token-loss \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --value-clip 0.2 \
  --kl-coef 0.0 \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr "$LR" \
  --lr-decay-style constant \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.30 \
  --sglang-max-running-requests 64 \
  --sglang-attention-backend flashinfer \
  --sglang-log-level warning \
  --sglang-context-length 2048 \
  --sglang-max-prefill-tokens 2048 \
  --sglang-max-total-tokens 65536 \
  --eval-prompt-data countdown "$DATA_DIR/test.jsonl" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-temperature 0.0 \
  --eval-top-p 1.0 \
  --eval-max-response-len 512 \
  --n-samples-per-eval-prompt 1 \
  --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES" \
  --train-env-vars "$TRAIN_ENV_JSON" \
  --use-tensorboard \
  "$@" 2>&1 | tee "$OUT/train.log"
