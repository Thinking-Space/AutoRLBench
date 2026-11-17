#!/bin/bash
# MiniMax-M2.5 8L: 8GPU RL 训练 (EP=8, TP=1, sglang EP=8)
# 裁剪到 8 层, 256 experts, ~60B total / ~11B active per forward
#
# 训推一致性监控已内置（mismatch metrics + logdiff 分位数）
#
# Key parameters (override via env vars):
#   NUM_ROLLOUT              default 5
#   ROLLOUT_BATCH_SIZE       default 16
#   N_SAMPLES_PER_PROMPT     default 2
#   MAX_TOKENS_PER_GPU       default 2048
#   SGLANG_MEM_FRACTION      default 0.45
#   TENSORBOARD_DIR          default exp_log/tensorboard/minimax_m2_8L
set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# ============================================================
# 路径配置
# ============================================================
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

export MODEL_PATH=${BASE_DIR}/models/MiniMax-M2.5-8L
export MEGATRON_MODEL_PATH=${BASE_DIR}/checkpoints/minimax-m2-8L-megatron
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

# 复制 modeling 文件到 HuggingFace cache
mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${MODEL_PATH}/configuration_minimax_m2.py \
   ${MODEL_PATH}/modeling_minimax_m2.py \
   /root/.cache/huggingface/modules/transformers_modules/

TRAIN_FILE="${BASE_DIR}/data/train/dapo-math-17k.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/minimax_m2_8L}

# ============================================================
# 可配置参数
# ============================================================
NUM_ROLLOUT=${NUM_ROLLOUT:-5}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.45}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-2048}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}

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
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax_m2.sh"

MODEL_ARGS_FAST=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  case "$a" in
    --no-masked-softmax-fusion) continue ;;
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

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="minimax_m2_8L_${RUN_TAG}"

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
  --rollout-max-response-len 1024
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --attention-backend flash
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
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
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests 64
  --sglang-ep-size 8
)

MONITORING_ARGS=(
  --get-mismatch-metrics
  --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
  --custom-config-path examples/train_infer_mismatch_helper/mis_monitor_only.yaml
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
echo "MiniMax-M2.5 8L: ${NUM_ROLLOUT} rollouts, EP=8, EXP=${EXP_NAME}"
echo "Monitoring: mismatch_metrics=ON (KL/ppl_ratio/chi2 + logdiff percentiles)"
echo "BASE_DIR: ${BASE_DIR}"
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
  "${MONITORING_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "MiniMax-M2.5 8L done! EXP=${EXP_NAME}"
