#!/bin/bash
# Long-Context RL Training Strategy Research
#
# Goal: Train MiniCPM5 MoE with rollout-max-response-len=62000 on 8×H100 80GB
# without OOM, while achieving raw_reward growth.
#
# Key strategies:
# 1. Context Parallelism (CP) to split long sequences across GPUs
# 2. Activation recomputation to reduce backward-pass memory
# 3. CP-aware truncation to handle oversized samples
# 4. Dynamic batch sizing with proper token budgets
# 5. Expandable segments to reduce CUDA memory fragmentation
#
# Usage:
#   bash scripts/community/run-longcontext-research.sh [--config LEVEL] [extra args...]
#
# Config levels:
#   1 - Conservative: CP=4, max-tokens=8192, recompute full, n-samples=4  (safest)
#   2 - Moderate:     CP=4, max-tokens=12288, recompute full, n-samples=8 (balanced)
#   3 - Aggressive:   CP=2, max-tokens=16384, recompute full, n-samples=8 (most efficient if fits)
#   4 - Baseline:     Original template command (for comparison/OOM diagnosis)

set -x

# ============================================================
# Parse config level
# ============================================================
CONFIG_LEVEL=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_LEVEL="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

echo "=========================================="
echo "Long-Context Research: CONFIG_LEVEL=${CONFIG_LEVEL}"
echo "=========================================="

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# NOTE: expandable_segments:True is incompatible with TorchMemorySaver
# (used for colocated offloading). Do NOT set it here.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Path Configuration
# ============================================================
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

# export MODEL_PATH=${BASE_DIR}/models/minicpmmoe
export MODEL_PATH=/user/lixin/models/job_182108.step_50000
export MEGATRON_MODEL_PATH=/user/lixin/models/job_182108.step_50000.miles_megatron 
export CKPT_PATH=/data/checkpoints 

export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

TRAIN_FILE="/user/lixin/datasets/miles/dapo-math-17k.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${BASE_DIR}/exp_log/tensorboard/longcontext_research

NUM_ROLLOUT=${NUM_ROLLOUT:-10}

# ============================================================
# Cleanup
# ============================================================
if [ "${SKIP_KILL:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  sleep 3
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  pkill -9 redis || true
else
  echo "SKIP_KILL=1: skip pkill/ray stop cleanup"
fi

set -ex

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minicpm5_moe.64k.sh"

MODEL_ARGS_FAST=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  case "$a" in
    --no-masked-softmax-fusion) continue ;;
    --no-rope-fusion) continue ;;
    --attention-softmax-in-fp32) continue ;;
    --moe-router-dtype) skip_next=1; continue ;;
    --moe-token-dispatcher-type) skip_next=1; continue ;;
    *) MODEL_ARGS_FAST+=("$a") ;;
  esac
done
MODEL_ARGS_FAST+=(
  --bf16
  --grad-reduce-in-bf16
  --main-grads-dtype bf16
  --moe-router-dtype fp32
)

# ============================================================
# Config-Level Dependent Parameters
# ============================================================
case "$CONFIG_LEVEL" in
  1)
    # Conservative: CP=4, effective_limit=65536 (no truncation needed)
    # n_samples=4 for lower memory pressure, recompute enabled
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  2)
    # Moderate: CP=4, effective_limit=65536, full 8 samples, recompute
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  3)
    # Aggressive: CP=2, effective_limit=32768, caps context at 32K
    CP_SIZE=2
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=31744
    ROLLOUT_MAX_CONTEXT_LEN=32768
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  4)
    # Baseline: Original template command (expected to OOM)
    CP_SIZE=2
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.60
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=0  # no context-len cap
    RECOMPUTE_ARGS=()
    ;;
  10)
    # n_samples=16, longtext CP=4: more samples per prompt to reduce zero-advantage
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=4
    N_SAMPLES_PER_PROMPT=16
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  11)
    # n_samples=32, longtext CP=4: even more samples to further reduce zero-advantage
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=2
    N_SAMPLES_PER_PROMPT=32
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  13)
    # rollout_batch=8, n_samples=16, gbs=64: more prompts + more samples per prompt
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=16
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  14)
    # rollout_batch=16, n_samples=8, gbs=64: many prompts, standard n_samples
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=16
    N_SAMPLES_PER_PROMPT=8
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  15)
    # rollout_batch=16, n_samples=16, gbs=64: large scale
    CP_SIZE=4
    MAX_TOKENS_PER_GPU=16384
    LOGPROBS_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_BATCH_SIZE=16
    N_SAMPLES_PER_PROMPT=16
    GLOBAL_BATCH_SIZE=64
    SGLANG_MEM_FRACTION=0.55
    ROLLOUT_MAX_RESPONSE_LEN=62000
    ROLLOUT_MAX_CONTEXT_LEN=65536
    RECOMPUTE_ARGS=(
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
    )
    ;;
  *)
    echo "Unknown CONFIG_LEVEL=${CONFIG_LEVEL}. Use 1-4, 10-15."
    exit 1
    ;;
  esac

# DP size = total_gpus / (TP * PP * CP)
DP_SIZE=$(( GPUS_PER_NODE / CP_SIZE ))
echo "CP_SIZE=${CP_SIZE}, DP_SIZE=${DP_SIZE}, MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}"
echo "Effective per-microbatch limit: $((MAX_TOKENS_PER_GPU * CP_SIZE)) tokens"
echo "Max per-GPU tokens per sample: $((ROLLOUT_MAX_RESPONSE_LEN / CP_SIZE + 1024))"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="longctx_cfg${CONFIG_LEVEL}_cp${CP_SIZE}_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 9999
)

ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-prompt-len 1024
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
)

if [ "${ROLLOUT_MAX_CONTEXT_LEN}" -gt 0 ]; then
  ROLLOUT_ARGS+=(--rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN})
fi

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 1
  --context-parallel-size ${CP_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-rollout-routing-replay
  --use-miles-router
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
)

WANDB_ARGS=(--use-tensorboard)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests 512
  --sglang-disable-shared-experts-fusion
)

mkdir -p ${CKPT_PATH}/${EXP_NAME}/dump_details
MISC_ARGS=(
  --dump-details ${CKPT_PATH}/${EXP_NAME}/dump_details
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --moe-token-dispatcher-type alltoall
  --recompute-loss-function
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "LONG-CONTEXT RESEARCH: CONFIG=${CONFIG_LEVEL}, CP=${CP_SIZE}, EXP=${EXP_NAME}"
echo "max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}, effective_limit=$((MAX_TOKENS_PER_GPU * CP_SIZE))"
echo "rollout: batch=${ROLLOUT_BATCH_SIZE}, n_samples=${N_SAMPLES_PER_PROMPT}, gbs=${GLOBAL_BATCH_SIZE}"
echo "response_len=${ROLLOUT_MAX_RESPONSE_LEN}, context_len=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "recompute: ${RECOMPUTE_ARGS[*]:-NONE}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  --start-rollout-id 0 \
  --skip-eval-before-train \
  "${MODEL_ARGS_FAST[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RECOMPUTE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "Long-context research done! CONFIG=${CONFIG_LEVEL}, EXP=${EXP_NAME}"
