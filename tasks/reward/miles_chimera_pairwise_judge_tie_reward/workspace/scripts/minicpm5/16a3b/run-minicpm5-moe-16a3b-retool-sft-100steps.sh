#!/bin/bash
# MiniCPM5 MoE 16A3B ReTool SFT, 100 training steps. Miles saves Megatron torch_dist checkpoints directly.

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLUSTER=${CLUSTER:-ningxia_h100}

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

BASE_DIR=${BASE_DIR:-/para_hdd/user/panyinxu/$PROJECT_NAME/$JOB_ID}
mkdir -p "$BASE_DIR"

export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
USE_WANDB=${USE_WANDB:-0}

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
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HF_MODEL_DIR=${HF_MODEL_DIR:-/user/panyinxu/models/job_362482_step_47000/}
HF_MODULES_DIR=${HF_MODULES_DIR:-/root/.cache/huggingface/modules}
HF_DYNAMIC_MODULE_NAME="$(basename "${HF_MODEL_DIR%/}" | sed 's/\./_dot_/g; s/-/_/g')"
HF_DYNAMIC_MODULE_DIR="${HF_MODULES_DIR}/transformers_modules/${HF_DYNAMIC_MODULE_NAME}"
mkdir -p "${HF_DYNAMIC_MODULE_DIR}"
touch "${HF_MODULES_DIR}/transformers_modules/__init__.py" "${HF_DYNAMIC_MODULE_DIR}/__init__.py"
for py_file in configuration_minicpm.py modeling_minicpm.py; do
  if [ -f "${HF_MODEL_DIR}/${py_file}" ]; then
    cp "${HF_MODEL_DIR}/${py_file}" "${HF_DYNAMIC_MODULE_DIR}/"
  fi
done

export MODEL_PATH=${HF_MODEL_DIR}
export MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-/user/panyinxu/ckpts/minicpm5_362482_step47000}
export CKPT_PATH=${CKPT_PATH:-${BASE_DIR}/checkpoints}
export PYTHONPATH=.:Megatron-LM/:mbridge:${HF_MODULES_DIR}
source "${WORK_DIR}/scripts/minicpm5/16a3b/minicpm5_moe_16a3b_rope.sh"

MODEL_ARGS_SELECTED=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  if [ "$a" = "--moe-router-dtype" ] || [ "$a" = "--moe-token-dispatcher-type" ]; then
    skip_next=1
    continue
  fi
  MODEL_ARGS_SELECTED+=("$a")
done
MODEL_ARGS_SELECTED+=(--bf16 --moe-router-dtype fp32)

MODEL_TAG=minicpm5_moe_16a3b
TASK_TAG=retool_sft_100steps

SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-/user/panyinxu/datasets/miles_data/retool-sft-multi-turn-train-2000.jsonl}
NUM_ROLLOUT=${NUM_ROLLOUT:-100}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16000}

export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/${MODEL_TAG}_${TASK_TAG}}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${MODEL_TAG}_${TASK_TAG}_${RUN_TAG}"
SAVE_PATH=${CKPT_PATH}/${EXP_NAME}/

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${SAVE_PATH}
  --save-interval ${SAVE_INTERVAL:-100}
  --no-save-optim
  --no-save-rng
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data $SFT_TRAIN_FILE
  --input-key ${INPUT_KEY:-messages}
  --tool-key ${TOOL_KEY:-tools}
  --rollout-shuffle
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --loss-type sft_loss
  --loss-mask-type ${LOSS_MASK_TYPE:-qwen}
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
  --expert-tensor-parallel-size 1
  --attention-backend ${ATTENTION_BACKEND:-flash}
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR:-1e-5}
  --lr-decay-style cosine
  --min-lr ${MIN_LR:-1e-6}
  --lr-warmup-fraction ${LR_WARMUP_FRACTION:-0.03}
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

WANDB_ARGS=(--use-tensorboard)
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(--use-wandb --wandb-project ${WANDB_PROJECT:-miles-minicpm5})
  if [ -n "${WANDB_TEAM:-}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
fi

mkdir -p ${SAVE_PATH}/dump_details
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${HF_MODULES_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG:-0}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL:-0}\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"NVTE_UNFUSED_ATTN\": \"0\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TMPDIR\": \"${TMPDIR}\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "${MODEL_TAG} ${TASK_TAG}: SFT"
echo "steps=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}"
echo "TP=${TENSOR_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, EP=${EXPERT_MODEL_PARALLEL_SIZE}"
echo "TRAIN_FILE=${SFT_TRAIN_FILE}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "After training, use this torch_dist checkpoint for RL:"
echo "  MEGATRON_MODEL_PATH=${SAVE_PATH}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  "${MODEL_ARGS_SELECTED[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "ReTool SFT done! torch_dist checkpoint: ${SAVE_PATH}"
