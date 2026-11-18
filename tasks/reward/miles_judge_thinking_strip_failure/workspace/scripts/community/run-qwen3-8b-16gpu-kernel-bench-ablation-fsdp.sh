#!/bin/bash
# Qwen3-8B - Kernel Bench RL Training — ABLATION FSDP (with ZMQ Sandbox)
# 2 nodes × 8 GPU = 16 GPU, FSDP training + SGLang rollout
#
# Ablation changes vs Megatron baseline:
#   1. Training backend: FSDP (was Megatron TP=4)
#   2. KL regularization: enabled (kl_loss_coef=0.01)
#   3. Clip ratio:        symmetric 0.2 / 0.2 (was 0.2 / 0.28)
#   4. Reward function:   more speedup-oriented (base=0.4, bonus_max=0.6, coef=0.3)
#
# Config:
#   Training:  16 GPU, FSDP DP=16, gradient checkpointing, FA3
#   Rollout:   sglang TP=2, 8 engines, FA3 backend
#   Data:      kernel_bench_data.jsonl (250 CUDA kernel generation tasks)
#   GRPO:      TIS+RS enabled, rm-type=triton (continuous reward)
#
# Key parameters (override via env vars):
#   NUM_ROLLOUT            default 200
#   ROLLOUT_BATCH_SIZE     default 8
#   N_SAMPLES_PER_PROMPT   default 16
#   MAX_TOKENS_PER_GPU     default 8192
#   MILES_MAX_RESPONSE_LEN default 32768
#   SANDBOX_PORT           default 5555

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CLUSTER=${CLUSTER:-ningxia_h100}
source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# ============================================================
# Path configuration
# ============================================================
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/models/qwen3-8b-sft}
export MODEL_PATH
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

TRAIN_FILE="${BASE_DIR}/data/train/kernel_bench_data.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/qwen3_8b_kernel_bench_ablation_fsdp}
USE_WANDB=${USE_WANDB:-0}

# ============================================================
# Sandbox configuration (ZMQ Triton Bench Server)
# ============================================================
SANDBOX_DIR="${BASE_DIR}/sandbox"
export SANDBOX_PORT=${SANDBOX_PORT:-5555}
export SANDBOX_HOST=${SANDBOX_HOST:-${MASTER_ADDR}}
export SANDBOX_ADDR="tcp://${SANDBOX_HOST}"
export TRITON_HOME=${TRITON_HOME:-${BASE_DIR}/triton_home}
export SANDBOX_READY_TIMEOUT=${SANDBOX_READY_TIMEOUT:-180}
mkdir -p "${TRITON_HOME}/torch_" "${TRITON_HOME}/triton_" "${TRITON_HOME}/content_"

# ============================================================
# Configurable parameters
# ============================================================
NUM_ROLLOUT=${NUM_ROLLOUT:-200}

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

# ============================================================
# Batch & memory configuration
# ============================================================
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-8192}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-1024}

ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.40}

MILES_MAX_PROMPT_LEN="${MILES_MAX_PROMPT_LEN:-4096}"
# FSDP actor backward OOMed on near-32k responses after several hours.
# Keep a safer default while still allowing long generations via env override.
MILES_MAX_RESPONSE_LEN="${MILES_MAX_RESPONSE_LEN:-24576}"

SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-40960}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-${SGLANG_CONTEXT_LEN}}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-1500000}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-64}
SGLANG_SCHEDULE_CONSERVATIVENESS=${SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}

RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-16000000000}
RAY_SPILL_MIN_FREE_PCT=${RAY_SPILL_MIN_FREE_PCT:-10}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="qwen3_8b_kernel_bench_ablation_fsdp_16gpu_${RUN_TAG}"

# ============================================================
# Args: checkpoint (HF format for FSDP)
# ============================================================
CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 10
)

# ============================================================
# Args: rollout
# ============================================================
ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key question
  --label-key final_answer
  --reward-key score
  --apply-chat-template
  --rollout-shuffle
  --rm-type triton
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-prompt-len ${MILES_MAX_PROMPT_LEN}
  --rollout-max-response-len ${MILES_MAX_RESPONSE_LEN}
  --rollout-temperature 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
  --use-miles-router
  --use-distributed-post
)

# ============================================================
# Args: FSDP training backend
# ============================================================
TRAIN_BACKEND_ARGS=(
  --train-backend fsdp
  --gradient-checkpointing
  --recompute-loss-function
  --attn-implementation flash_attention_3
  --update-weight-buffer-size 536870912
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

# ============================================================
# Args: GRPO (ablation: KL=0.01, symmetric clip)
# ============================================================
GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
)

# MIS config: TIS + RS
MIS_CONFIG_PATH="${CKPT_PATH}/${EXP_NAME}/mis_config.yaml"
mkdir -p "$(dirname "${MIS_CONFIG_PATH}")"
cat > "${MIS_CONFIG_PATH}" <<'MISEOF'
use_tis: true
use_rs: true
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
MISEOF

MIS_ARGS=(
  --custom-config-path "${MIS_CONFIG_PATH}"
  --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# ============================================================
# Args: optimizer
# ============================================================
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --lr-warmup-iters 15
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

# ============================================================
# Args: logging
# ============================================================
WANDB_ARGS=(--use-tensorboard)
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT:-miles-qwen3-8b}
  )
  if [ -n "${WANDB_TEAM:-}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
fi

# ============================================================
# Args: SGLang (TP=2, 8 engines on 16 GPUs)
# ============================================================
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
  --sglang-schedule-conservativeness ${SGLANG_SCHEDULE_CONSERVATIVENESS}

  --sglang-attention-backend fa3
  --sglang-context-length ${SGLANG_CONTEXT_LEN}
  --sglang-max-prefill-tokens ${SGLANG_MAX_PREFILL_TOKENS}
  --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}

  --sglang-enable-metrics
)

# ============================================================
# Args: misc
# ============================================================
mkdir -p ${CKPT_PATH}/${EXP_NAME}/dump_details
MISC_ARGS=(
  --dump-details ${CKPT_PATH}/${EXP_NAME}/dump_details
)

# ============================================================
# Runtime env (with ablation reward env vars)
# ============================================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"NVTE_UNFUSED_ATTN\": \"0\",
    \"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION\": \"10000\",
    \"TORCHDYNAMO_DISABLE\": \"1\",
    \"SANDBOX_ADDR\": \"${SANDBOX_ADDR}\",
    \"SANDBOX_PORT\": \"${SANDBOX_PORT}\",
    \"TRITON_HOME\": \"${TRITON_HOME}\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "Qwen3-8B Kernel Bench ABLATION FSDP (16 GPU): ${NUM_ROLLOUT} rollouts"
echo "  FSDP DP=16, gradient_checkpointing=on, FA3"
echo "  samples=${N_SAMPLES_PER_PROMPT}, batch=${ROLLOUT_BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}"
echo "  prompt_max=${MILES_MAX_PROMPT_LEN}, response_max=${MILES_MAX_RESPONSE_LEN}"
echo "  sglang: TP=${ROLLOUT_NUM_GPUS_PER_ENGINE}, engines=$((16 / ROLLOUT_NUM_GPUS_PER_ENGINE)), mem=${SGLANG_MEM_FRACTION}"
echo "  train: max_tokens=${MAX_TOKENS_PER_GPU}, logprobs_max_tokens=${LOGPROBS_MAX_TOKENS_PER_GPU}, logprobs_chunk=${LOG_PROBS_CHUNK_SIZE}"
echo "  ray: object_store=${RAY_OBJECT_STORE_MEMORY}, spill_min_free_pct=${RAY_SPILL_MIN_FREE_PCT}"
echo "  KL=off, clip=0.2/0.28, reward=binary"
echo "  TIS=on, RS=on, FA3=on, rm_type=triton (sandbox)"
echo "  miles_router=on, distributed_post=on, recompute_loss=on"
echo "  sandbox: ${SANDBOX_ADDR}:${SANDBOX_PORT}"
echo "  EXP=${EXP_NAME}"
echo "  BASE_DIR: ${BASE_DIR}"
echo "============================================================"

# ---- Start ZMQ sandbox server and wait for ready ----
SANDBOX_LOG="${CKPT_PATH}/${EXP_NAME}/sandbox.log"
mkdir -p "$(dirname "${SANDBOX_LOG}")"
echo "Starting sandbox server on port ${SANDBOX_PORT} ..."
cd "${SANDBOX_DIR}" && python run_server.py --port "${SANDBOX_PORT}" > "${SANDBOX_LOG}" 2>&1 &
SANDBOX_PID=$!
cd "${WORK_DIR}"
echo "sandbox pid=${SANDBOX_PID}, log=${SANDBOX_LOG}"

python3 -c "
import os, socket, time, sys
host = os.environ['SANDBOX_HOST']
port = int(os.environ['SANDBOX_PORT'])
timeout = int(os.environ.get('SANDBOX_READY_TIMEOUT', '180'))
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        break
    except (ConnectionRefusedError, OSError):
        time.sleep(2)
else:
    print(f'ERROR: sandbox {host}:{port} not ready within {timeout}s')
    sys.exit(1)
print(f'sandbox ready: {host}:{port}')
"
echo "Sandbox server ready."

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  --start-rollout-id 0 \
  --skip-eval-before-train \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${MIS_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${TRAIN_BACKEND_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh

# Clean up sandbox
if [ -n "${SANDBOX_PID:-}" ] && kill -0 "${SANDBOX_PID}" 2>/dev/null; then
  echo "Stopping sandbox server (pid=${SANDBOX_PID}) ..."
  kill "${SANDBOX_PID}" 2>/dev/null || true
fi
echo "Qwen3-8B Kernel Bench ABLATION FSDP (16 GPU) done! EXP=${EXP_NAME}"
