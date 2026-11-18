#!/bin/bash
# Test script: reproduce OOM reported with --rollout-max-response-len 64000
# and verify the chunked-logits fix in forward_only works correctly.
#
# Key parameter overrides from bug report:
#   --rollout-batch-size 32  --n-samples-per-prompt 16  --global-batch-size 512
#   --rollout-max-response-len 64000  --log-probs-chunk-size 4096

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

export MODEL_PATH=${BASE_DIR}/models/minicpmmoe
export MEGATRON_MODEL_PATH=${BASE_DIR}/checkpoints/minicpm5-moe-megatron-fixed-v2
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

TRAIN_FILE="${BASE_DIR}/data/train/dapo-math-17k.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${BASE_DIR}/exp_log/tensorboard/test_oom_fix

NUM_ROLLOUT=1

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

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
NVLINK_COUNT=$(nvidia-smi nvlink -s 2>/dev/null | grep -c "Link" || echo "0")
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo "1" || echo "0")
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minicpm5_moe.sh"

MODEL_ARGS_FAST=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ $skip_next -eq 1 ]; then skip_next=0; continue; fi
  case "$a" in
    --use-fused-rmsnorm|--use-fused-rotary-pos-emb|--use-fused-swiglu) continue ;;
    --params-dtype) skip_next=1; continue ;;
  esac
  MODEL_ARGS_FAST+=("$a")
done
MODEL_ARGS_FAST+=(
  --bf16
  --moe-router-dtype fp32
)

# === Bug report parameters ===
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=512

MAX_TOKENS_PER_GPU=12288
LOGPROBS_MAX_TOKENS_PER_GPU=16384
SGLANG_MEM_FRACTION=0.60

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="test_oom_fix_${RUN_TAG}"

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
  --rollout-max-response-len 64000
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
  --log-probs-chunk-size 4096
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
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
echo "TEST OOM FIX: rollout-max-response-len=64000, log-probs-chunk-size=4096"
echo "EXP=${EXP_NAME}"
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
  "${MISC_ARGS[@]}"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "TEST OOM FIX done! EXP=${EXP_NAME}"
