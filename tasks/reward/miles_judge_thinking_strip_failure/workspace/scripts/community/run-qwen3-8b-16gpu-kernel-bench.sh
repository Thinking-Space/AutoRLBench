#!/bin/bash
# ============================================================
# Qwen3-8B Kernel Bench RL Training (16 GPU, multi-node ZMQ sandbox)
# 2 nodes x 8 GPU = 16 GPU
#
# Restored 2026-04-18 after the working copy was unexpectedly removed
# from disk between exp02's submission (04-18 03:59) and the next
# triage at 08:06. This file is reverse-engineered from the `set -x`
# trace of Job 263430 (exp01) master+worker logs, so it should be
# byte-equivalent in behavior to the version exp01/exp02 actually ran.
#
# Key fix vs the 2026-04-15 broken sandbox script
# -----------------------------------------------
#   Sandbox discovery dir was previously
#       ${SAVE_DIR}/sandbox_discovery
#   which depended on ${SAVE_DIR}=${EXP_NAME_PREFIX}_${RUN_TAG}; rank 0
#   and rank 1 captured `date +%Y%m%d_%H%M%S` at slightly different
#   wall-clock seconds, so they wrote to different directories and the
#   master never collected rank 1's endpoint -> SANDBOX_ENDPOINTS empty
#   -> all sandbox traffic hammered the master node -> 819 ZMQ timeouts/day.
#   This version uses
#       ${CKPT_PATH}/sandbox_discovery_${MASTER_ADDR}
#   which is identical across ranks (MASTER_ADDR is injected by the
#   PyTorchJob controller) and is cleaned up by rank 0 right after
#   aggregation.
#
# Other deltas vs that broken version
# -----------------------------------
#   - Removed --over-sampling-batch-size (caused sandbox to be hit 2x)
#   - Removed --dynamic-sampling-filter-path  (was discarding all the
#     easy-injection solve-all groups; the GRPO-side
#     --grpo-filter-groups-with-same-reward is kept since that one
#     prunes the gradient-zero groups inside training, not by resampling)
#   - Removed --dynamic-sampling-resample-same-prompt-max-times
#   - dynamic_sampling banner flipped on -> off accordingly
#
# Architecture
# ------------
#   Train: TP=4, DP=4, CP=1, dense Qwen3-8B (full attention, sequence-parallel)
#   Rollout: sglang TP=4, 4 engines (1 per DP rank), FA3 backend
#   Reward: ZMQ Triton sandbox, one server per node, master aggregates
#           endpoints into SANDBOX_ENDPOINTS so rollout actors fan out
#           ZMQ requests across all sandboxes (round-robin).
#
# Tunables (env-var driven)
# -------------------------
#   NUM_ROLLOUT             default 200
#   ROLLOUT_BATCH_SIZE      default 16
#   N_SAMPLES_PER_PROMPT    default 16
#   ROLLOUT_TEMPERATURE     default 0.6
#   ENTROPY_COEF            default 0.01
#   KL_LOSS_COEF            default 0.005
#   LR                      default 2e-6
#   LR_WARMUP_ITERS         default 30
#   MILES_MAX_PROMPT_LEN    default 4096
#   MILES_MAX_RESPONSE_LEN  default 16384
#   TENSOR_PARALLEL_SIZE    default 4
#   ROLLOUT_NUM_GPUS_PER_ENGINE default 4
#   SANDBOX_PORT            default 5555
#   SANDBOX_MAX_QUEUE       default 256
#   SANDBOX_TIMEOUT_MS      default 600000
#   SAVE_INTERVAL           default 10
#   EXP_NAME_PREFIX         default qwen3_8b_kernel_bench_16gpu
#   RESUME_FROM             default ""    (path to a megatron ckpt dir)
#   RESUME_EXPECTED_ITER    default ""    (start-rollout-id when resuming)
# ============================================================

set -ex
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CLUSTER=${CLUSTER:-paratera_shandong}
source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# ============================================================
# Path configuration
# ============================================================
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/models/qwen3-8b-sft}
export MODEL_PATH
export MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${BASE_DIR}/checkpoints/qwen3-8b-sft-megatron}
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

TRAIN_FILE=${TRAIN_FILE:-${BASE_DIR}/data/train/kernel_bench_dataL1L2.jsonl}
# Note: use ${VAR-default} (no colon) so an explicit empty string is preserved.
# This lets exp scripts opt out of the easy-injection / extra train file
# entirely (e.g. exp13 ablations need a single-source dataset).
EASY_TRAIN_FILE=${EASY_TRAIN_FILE-${BASE_DIR}/data/train/kernel_bench_easy_injection.jsonl}
EXTRA_TRAIN_FILE=${EXTRA_TRAIN_FILE-}
TRAIN_FILES=("${TRAIN_FILE}")
if [ -n "${EASY_TRAIN_FILE}" ]; then
  TRAIN_FILES+=("${EASY_TRAIN_FILE}")
fi
if [ -n "${EXTRA_TRAIN_FILE}" ]; then
  TRAIN_FILES+=("${EXTRA_TRAIN_FILE}")
fi

export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
USE_WANDB=${USE_WANDB:-0}
export MILES_DISABLE_LOGDIFF_METRICS=${MILES_DISABLE_LOGDIFF_METRICS:-1}

# ============================================================
# Sandbox configuration (ZMQ Triton Bench Server, one per node)
# ============================================================
SANDBOX_DIR=${SANDBOX_DIR:-${BASE_DIR}/sandbox}
export SANDBOX_PORT=${SANDBOX_PORT:-5555}
export SANDBOX_HOST=${SANDBOX_HOST:-${MASTER_ADDR}}
export SANDBOX_ADDR=${SANDBOX_ADDR:-tcp://${SANDBOX_HOST}}
export SANDBOX_MAX_QUEUE=${SANDBOX_MAX_QUEUE:-256}
export SANDBOX_TIMEOUT_MS=${SANDBOX_TIMEOUT_MS:-600000}
# TRITON_HOME holds short-lived per-sample temp files (content_*.txt /
# triton_*.py / torch_*.py) created by miles/rollout/rm_hub/triton_verify.py
# in compute_score(). They are os.remove()'d in the same call's finally:
# block, so they have second-scale lifetime and never cross node boundaries
# (sandbox traffic is in-memory ZMQ, not file-based). Keep them on the Pod's
# local NVMe under /data instead of the shared GPFS to avoid (1) disk-full
# blast radius hitting all jobs at once, and (2) the hot-path latency of
# writing a few-KB file to the shared FS for every reward computation.
# Same naming convention as RAY_TMPDIR / TORCH_EXTENSIONS_DIR in env.sh.
export TRITON_HOME=${TRITON_HOME:-/data/logs/tmp/triton_home/rank_${RANK:-0}}
export SANDBOX_READY_TIMEOUT=${SANDBOX_READY_TIMEOUT:-180}
mkdir -p "${TRITON_HOME}/torch_" "${TRITON_HOME}/triton_" "${TRITON_HOME}/content_"

NUM_ROLLOUT=${NUM_ROLLOUT:-200}

# ============================================================
# TIMING instrumentation (TB scalars + text under script_timing/*)
# ============================================================
TB_TIMING_LOGDIR=${TENSORBOARD_DIR}
TB_TIMING_STEP=${TB_TIMING_STEP:-$(date +%s)}
TB_TIMING_ENABLED=0
init_tb_timing() {
  if [ -z "${TB_TIMING_LOGDIR}" ]; then
    return 0
  fi
  if python3 - "${TB_TIMING_LOGDIR}" <<'PY' 2>/dev/null
import sys
from torch.utils.tensorboard import SummaryWriter
SummaryWriter(sys.argv[1]).close()
PY
  then
    TB_TIMING_ENABLED=1
    echo "[TIMING] tensorboard timing logdir: ${TB_TIMING_LOGDIR}"
  else
    echo "[TIMING] tensorboard not available, scalar logging disabled"
  fi
}
log_tb_scalar() {
  local tag=$1 value=$2 step=${3:-${TB_TIMING_STEP}}
  if [ "${TB_TIMING_ENABLED}" = "1" ]; then
    python3 - "${TB_TIMING_LOGDIR}" "${tag}" "${value}" "${step}" <<'PY' 2>/dev/null || true
import sys
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter(sys.argv[1])
w.add_scalar(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
w.flush(); w.close()
PY
  fi
}
log_tb_text() {
  local tag=$1 text=$2 step=${3:-${TB_TIMING_STEP}}
  if [ "${TB_TIMING_ENABLED}" = "1" ]; then
    TIMING_TEXT="${text}" python3 - "${TB_TIMING_LOGDIR}" "${tag}" "${step}" <<'PY' 2>/dev/null || true
import os, sys
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter(sys.argv[1])
w.add_text(sys.argv[2], os.environ.get('TIMING_TEXT', ''), int(sys.argv[3]))
w.flush(); w.close()
PY
  fi
}
format_duration() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02dh:%02dm:%02ds" "${hours}" "${minutes}" "${seconds}"
}
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
phase_start_human() { local v="${1}_START_HUMAN"; printf %s "${!v}"; }
phase_end_human()   { local v="${1}_END_HUMAN";   printf %s "${!v}"; }
log_tb_phase_metrics() {
  local phase=$1 duration_seconds=$2 exit_code=$3
  log_tb_scalar "script_timing/duration_seconds/${phase}" "${duration_seconds}"
  log_tb_scalar "script_timing/exit_code/${phase}"        "${exit_code}"
}
start_phase() {
  local phase=$1
  local now_ts
  now_ts=$(date +%s)
  printf -v "${phase}_START_TS"    %s "${now_ts}"
  printf -v "${phase}_START_HUMAN" %s "$(timestamp)"
  echo "[TIMING] ${phase} start: $(phase_start_human ${phase})"
}
end_phase() {
  local phase=$1 exit_code=$2
  local start_var="${phase}_START_TS"
  local start_ts="${!start_var}"
  if [ -z "${start_ts}" ]; then
    return 0
  fi
  local end_ts duration_seconds
  end_ts=$(date +%s)
  duration_seconds=$((end_ts - start_ts))
  printf -v "${phase}_END_TS"            %s "${end_ts}"
  printf -v "${phase}_END_HUMAN"         %s "$(timestamp)"
  printf -v "${phase}_DURATION_SECONDS"  %s "${duration_seconds}"
  printf -v "${phase}_EXIT_CODE"         %s "${exit_code}"
  echo "[TIMING] ${phase} end: $(phase_end_human ${phase}) elapsed=$(format_duration ${duration_seconds}) exit_code=${exit_code}"
  log_tb_phase_metrics "${phase}" "${duration_seconds}" "${exit_code}"
}

# ============================================================
# preflight cleanup
# ============================================================
start_phase preflight_cleanup
if [ "${SKIP_CLEANUP:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -f 'python3 train.py' || true
  sleep 3
  pkill -9 ray || true
  pkill -f 'python3 train.py' || true
  pkill -9 redis || true
fi
end_phase preflight_cleanup 0

set -ex
export PYTHONBUFFERED=16

# ============================================================
# config_prepare
# ============================================================
start_phase config_prepare

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ ${NVLINK_COUNT} -gt 0 ]] && echo 1 || echo 0)
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

MODEL_ARGS+=(
  --bf16
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
)

# ---- parallelism ----
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

# ---- batch ----
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

# ---- rollout / GRPO knobs ----
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
ENTROPY_COEF=${ENTROPY_COEF:-0.01}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
LR=${LR:-2e-6}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-30}
# LR scheduler: default constant (back-compat). Set to cosine/linear/inverse-square-root
# to enable decay; LR_DECAY_ITERS / MIN_LR are only forwarded when style != constant.
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-${NUM_ROLLOUT}}
MIN_LR=${MIN_LR:-0}

# ---- early-stop on no_grad (avoid wasting compute on dead policy) ----
NO_GRAD_STOP_THRESHOLD=${NO_GRAD_STOP_THRESHOLD:-0.8}
NO_GRAD_STOP_PATIENCE=${NO_GRAD_STOP_PATIENCE:-3}

# ---- compute / memory ----
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-10240}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-10240}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-1024}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.60}

# ---- lengths ----
MILES_MAX_PROMPT_LEN=${MILES_MAX_PROMPT_LEN:-4096}
MILES_MAX_RESPONSE_LEN=${MILES_MAX_RESPONSE_LEN:-16384}

# ---- sglang ----
SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-40960}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-${SGLANG_CONTEXT_LEN}}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-1500000}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-256}
SGLANG_SCHEDULE_CONSERVATIVENESS=${SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}

# ---- ray ----
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-16000000000}
RAY_SPILL_MIN_FREE_PCT=${RAY_SPILL_MIN_FREE_PCT:-10}
export RAY_OBJECT_STORE_MEMORY RAY_SPILL_MIN_FREE_PCT

# ---- resume ----
RESUME_FROM=${RESUME_FROM:-}
RESUME_EXPECTED_ITER=${RESUME_EXPECTED_ITER:-}

# ---- naming ----
RUN_TAG=$(date +%Y%m%d_%H%M%S)
EXP_NAME_PREFIX=${EXP_NAME_PREFIX:-qwen3_8b_kernel_bench_16gpu}
EXP_NAME="${EXP_NAME_PREFIX}_${RUN_TAG}"
SAVE_DIR=${CKPT_PATH}/${EXP_NAME}
mkdir -p "${SAVE_DIR}"
mkdir -p "${SAVE_DIR}/dump_details"

# ---- resume args ----
CKPT_LOAD_PATH=${MEGATRON_MODEL_PATH}
RESUME_ARGS=(--start-rollout-id 0)
if [ -n "${RESUME_FROM}" ]; then
  CKPT_LOAD_PATH="${RESUME_FROM}"
  RESUME_ARGS=(--start-rollout-id "${RESUME_EXPECTED_ITER:-0}")
fi

# ---- TB metadata (rank 0 only; on workers TB_TIMING_LOGDIR may differ) ----
init_tb_timing
log_tb_scalar script_timing/run_status/started 1
log_tb_text script_timing/run_metadata "exp_name: ${EXP_NAME}
run_tag: ${RUN_TAG}
train_file: ${TRAIN_FILES[*]}
resume_from: ${RESUME_FROM:-none}
resume_saved_iter: ${RESUME_EXPECTED_ITER:-fresh}
tensorboard_dir: ${TENSORBOARD_DIR}
sandbox_addr: ${SANDBOX_ADDR}:${SANDBOX_PORT}
num_rollout: ${NUM_ROLLOUT}
rollout_batch_size: ${ROLLOUT_BATCH_SIZE}
n_samples_per_prompt: ${N_SAMPLES_PER_PROMPT}"

# ============================================================
# arg arrays
# ============================================================
CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load ${CKPT_LOAD_PATH}
  --save ${SAVE_DIR}/
  --save-interval ${SAVE_INTERVAL:-10}
  --async-save
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_FILES[@]}"
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
  --rollout-temperature ${ROLLOUT_TEMPERATURE}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
  --use-miles-router
  --use-distributed-post
)
if [ -n "${NO_GRAD_STOP_THRESHOLD}" ]; then
  ROLLOUT_ARGS+=(--stop-on-no-grad-threshold "${NO_GRAD_STOP_THRESHOLD}")
fi
if [ -n "${NO_GRAD_STOP_PATIENCE}" ]; then
  ROLLOUT_ARGS+=(--stop-on-no-grad-patience "${NO_GRAD_STOP_PATIENCE}")
fi
# Optional dynamic-sampling hooks (default OFF; exp06+ enables them to filter
# all_one / all_zero groups before training, directly attacking the signal
# collapse seen in exp04 step50+ where all_one jumped 7/16 -> 9/16).
if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [ -n "${OVER_SAMPLING_BATCH_SIZE}" ]; then
  ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [ -n "${DYNAMIC_SAMPLING_RESAMPLE_SAME_PROMPT_MAX_TIMES}" ]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-resample-same-prompt-max-times "${DYNAMIC_SAMPLING_RESAMPLE_SAME_PROMPT_MAX_TIMES}")
fi

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --attention-backend ${ATTENTION_BACKEND:-flash}
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

GRPO_ARGS=(
  --advantage-estimator ${ADVANTAGE_ESTIMATOR:-grpo}
  --use-kl-loss
  --kl-loss-coef ${KL_LOSS_COEF}
  --kl-loss-type low_var_kl
  --entropy-coef ${ENTROPY_COEF}
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
  --grpo-filter-groups-with-same-reward
  --ref-load $MEGATRON_MODEL_PATH
)

if [ "${ENABLE_PARTIAL_ROLLOUT:-0}" = "1" ]; then
  GRPO_ARGS+=(--partial-rollout)
fi
if [ "${ENABLE_PARTIAL_OFFPOLICY_MASK:-0}" = "1" ]; then
  GRPO_ARGS+=(--mask-offpolicy-in-partial-rollout)
fi

MIS_CONFIG_PATH=${SAVE_DIR}/mis_config.yaml
mkdir -p "$(dirname "${MIS_CONFIG_PATH}")"
cat > "${MIS_CONFIG_PATH}" <<'EOF'
use_tis: true
use_rs: false
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
EOF
MIS_ARGS=(
  --custom-config-path "${MIS_CONFIG_PATH}"
  --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR}
  --lr-decay-style ${LR_DECAY_STYLE}
  --lr-warmup-iters ${LR_WARMUP_ITERS}
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)
if [ "${LR_DECAY_STYLE}" != "constant" ]; then
  OPTIMIZER_ARGS+=(
    --lr-decay-iters ${LR_DECAY_ITERS}
    --min-lr ${MIN_LR}
  )
fi

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

MISC_ARGS=(
  --dump-details ${SAVE_DIR}/dump_details
  --attention-dropout 0.0
  --hidden-dropout 0.0
)

end_phase config_prepare 0

# ============================================================
# sandbox_total
#   - every rank starts a local sandbox + writes its endpoint to
#     ${SANDBOX_DISCOVERY_DIR}/rank_${RANK}
#   - rank 0 starts ray head, waits for all ranks' endpoint files,
#     aggregates them into SANDBOX_ENDPOINTS, and waits for its own
#     local sandbox to be ready before submitting train.py
#   - rank > 0 blocks inside ray_start.sh (--block) until master
#     stops the cluster
# ============================================================
start_phase sandbox_total

SANDBOX_LOG_DIR=${SAVE_DIR}/sandbox_logs
mkdir -p "${SANDBOX_LOG_DIR}"
LOCAL_SANDBOX_LOG=${SANDBOX_LOG_DIR}/sandbox_rank${RANK}.log

start_phase sandbox_spawn
echo "Starting sandbox server on port ${SANDBOX_PORT} (rank ${RANK}) ..."
(
  cd "${SANDBOX_DIR}" && python run_server.py --port "${SANDBOX_PORT}" --max-queue "${SANDBOX_MAX_QUEUE}"
) > "${LOCAL_SANDBOX_LOG}" 2>&1 &
SANDBOX_PID=$!
end_phase sandbox_spawn 0
echo "sandbox pid=${SANDBOX_PID}, log=${LOCAL_SANDBOX_LOG}"

LOCAL_IP=$(hostname -i | awk '{print $1}')
SANDBOX_DISCOVERY_DIR=${CKPT_PATH}/sandbox_discovery_${MASTER_ADDR}
mkdir -p "${SANDBOX_DISCOVERY_DIR}"
echo "tcp://${LOCAL_IP}:${SANDBOX_PORT}" > "${SANDBOX_DISCOVERY_DIR}/rank_${RANK}"

start_phase ray_cluster_start
set +e
bash scripts/setup/ray_start.sh
RAY_START_EXIT_CODE=$?
set -e
end_phase ray_cluster_start ${RAY_START_EXIT_CODE}
if [ ${RAY_START_EXIT_CODE} -ne 0 ]; then
  echo "ERROR: ray_start.sh exited ${RAY_START_EXIT_CODE}"
  exit ${RAY_START_EXIT_CODE}
fi

# rank > 0 ray_start.sh has --block and never returns. If we somehow
# get here on a worker (cluster shutdown), exit cleanly.
if [ "${RANK}" != "0" ]; then
  echo "RANK ${RANK}: ray_start.sh returned (cluster shutdown). Exiting."
  if [ -n "${SANDBOX_PID:-}" ] && kill -0 "${SANDBOX_PID}" 2>/dev/null; then
    kill "${SANDBOX_PID}" 2>/dev/null || true
  fi
  exit 0
fi

echo "Waiting for sandbox discovery files from all ${WORLD_SIZE} nodes..."
MAX_SANDBOX_WAIT=${MAX_SANDBOX_WAIT:-120}
SANDBOX_WAITED=0
while [ ${SANDBOX_WAITED} -lt ${MAX_SANDBOX_WAIT} ]; do
  DISCOVERED_COUNT=$(ls "${SANDBOX_DISCOVERY_DIR}"/rank_* 2>/dev/null | wc -l)
  if [ "${DISCOVERED_COUNT}" -ge "${WORLD_SIZE}" ]; then
    break
  fi
  sleep 2
  SANDBOX_WAITED=$((SANDBOX_WAITED + 2))
done

SANDBOX_ENDPOINTS=
for f in "${SANDBOX_DISCOVERY_DIR}"/rank_*; do
  ep=$(cat "${f}")
  if [ -n "${ep}" ]; then
    if [ -n "${SANDBOX_ENDPOINTS}" ]; then
      SANDBOX_ENDPOINTS="${SANDBOX_ENDPOINTS},${ep}"
    else
      SANDBOX_ENDPOINTS="${ep}"
    fi
  fi
done
export SANDBOX_ENDPOINTS
rm -rf "${SANDBOX_DISCOVERY_DIR}"
echo "SANDBOX_ENDPOINTS=${SANDBOX_ENDPOINTS}"

start_phase sandbox_wait_ready
set +e
python3 -c "
import os, socket, time, sys
host = os.environ.get('SANDBOX_HOST', 'localhost')
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
SANDBOX_READY_EXIT_CODE=$?
set -e
end_phase sandbox_wait_ready ${SANDBOX_READY_EXIT_CODE}

end_phase sandbox_total 0

if [ ${SANDBOX_READY_EXIT_CODE} -ne 0 ]; then
  echo "ERROR: sandbox not ready, aborting"
  exit ${SANDBOX_READY_EXIT_CODE}
fi

# ---- runtime env JSON for ray job submit (built AFTER SANDBOX_ENDPOINTS is known) ----
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
    \"MILES_DISABLE_LOGDIFF_METRICS\": \"${MILES_DISABLE_LOGDIFF_METRICS}\",
    \"SANDBOX_ADDR\": \"${SANDBOX_ADDR}\",
    \"SANDBOX_PORT\": \"${SANDBOX_PORT}\",
    \"SANDBOX_ENDPOINTS\": \"${SANDBOX_ENDPOINTS}\",
    \"SANDBOX_TIMEOUT_MS\": \"${SANDBOX_TIMEOUT_MS}\",
    \"TRITON_HOME\": \"${TRITON_HOME}\"
  }
}"

echo "Sandbox ready. Endpoints: ${SANDBOX_ENDPOINTS}"
echo "============================================================"
echo "Qwen3-8B Kernel Bench (16 GPU): ${NUM_ROLLOUT} rollouts"
echo "  TP=${TENSOR_PARALLEL_SIZE}, DP=$((16 / TENSOR_PARALLEL_SIZE)), CP=${CONTEXT_PARALLEL_SIZE}"
echo "  samples=${N_SAMPLES_PER_PROMPT}, batch=${ROLLOUT_BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}"
echo "  prompt_max=${MILES_MAX_PROMPT_LEN}, response_max=${MILES_MAX_RESPONSE_LEN}"
echo "  sglang: TP=${ROLLOUT_NUM_GPUS_PER_ENGINE}, engines=$((16 / ROLLOUT_NUM_GPUS_PER_ENGINE)), mem=${SGLANG_MEM_FRACTION}"
echo "  sglang: max_running=${SGLANG_MAX_RUNNING_REQUESTS}, schedule_conserv=${SGLANG_SCHEDULE_CONSERVATIVENESS}, max_total=${SGLANG_MAX_TOTAL_TOKENS}"
echo "  ray: object_store=${RAY_OBJECT_STORE_MEMORY}, spill_min_free_pct=${RAY_SPILL_MIN_FREE_PCT}"
echo "  TIS=on, RS=off, FA3=on, rm_type=triton (sandbox), dynamic_sampling=off"
echo "  miles_router=on, distributed_post=on"
echo "  resume_from: ${RESUME_FROM:-none} (saved_iter=${RESUME_EXPECTED_ITER:-fresh})"
echo "  sandbox: ${SANDBOX_ENDPOINTS}"
echo "  logdiff metrics: disabled=${MILES_DISABLE_LOGDIFF_METRICS}"
echo "  tensorboard: ${TENSORBOARD_DIR} (script_timing/*)"
echo "  EXP=${EXP_NAME}"
echo "  save_dir: ${SAVE_DIR}"
echo "  BASE_DIR: ${BASE_DIR}"
echo "============================================================"

start_phase train_submit_and_wait
set +e
ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  "${RESUME_ARGS[@]}" \
  --skip-eval-before-train \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${MIS_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"
TRAIN_EXIT_CODE=$?
set -e
end_phase train_submit_and_wait ${TRAIN_EXIT_CODE}

# ---- cleanup ----
bash $WORK_DIR/scripts/setup/ray_end.sh || true

if [ -n "${SANDBOX_PID:-}" ] && kill -0 "${SANDBOX_PID}" 2>/dev/null; then
  echo "Stopping sandbox server (pid=${SANDBOX_PID}) ..."
  kill "${SANDBOX_PID}" 2>/dev/null || true
fi

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
  log_tb_scalar script_timing/run_status/exit_code "${TRAIN_EXIT_CODE}"
  exit "${TRAIN_EXIT_CODE}"
fi

log_tb_scalar script_timing/run_status/completed 1
echo "Qwen3-8B Kernel Bench (16 GPU) done! EXP=${EXP_NAME}"
