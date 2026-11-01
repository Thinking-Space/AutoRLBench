#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"
CONDA_ENV_NAME="${BASELINE_CONDA_ENV:-${CONDA_ENV:-skillrl}}"
if [[ -n "$CONDA_ENV_NAME" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    set -u
fi

MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ezpoints_episode_skill_pipeline_qwen25_vl_3b_gemini_self_1440}"
BASELINE_ROLLOUTS_JSONL="${BASELINE_ROLLOUTS_JSONL:-$OUTPUT_DIR/baseline_rollouts.jsonl}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-60004}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://${HOST}:${PORT}/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-$MODEL_NAME}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.0}"
POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-512}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-120}"
POLICY_RETRIES="${POLICY_RETRIES:-2}"
POLICY_RETRY_DELAY="${POLICY_RETRY_DELAY:-1.0}"

NUM_TASKS="${NUM_TASKS:-180}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-180}"
REQUEST_WORKERS="${REQUEST_WORKERS:-64}"
MAX_STEPS="${MAX_STEPS:-8}"
SEED="${SEED:-2026}"
RESUME="${RESUME:-true}"
OVERWRITE="${OVERWRITE:-false}"

START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/baseline_policy_vllm.log}"

server_pid=""
server_started=0

is_server_ready() {
    curl -fsS "$POLICY_BASE_URL/models" >/dev/null 2>&1
}

stop_server() {
    if [[ "$server_started" != "1" || -z "$server_pid" || "$KEEP_VLLM_ALIVE" == "1" ]]; then
        return
    fi
    kill -TERM -- "-$server_pid" >/dev/null 2>&1 || kill -TERM "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" 2>/dev/null || true
}

trap stop_server EXIT INT TERM

if is_server_ready; then
    echo "Using existing visual policy server at $POLICY_BASE_URL"
elif [[ "$START_VLLM" == "1" ]]; then
    echo "Starting visual policy vLLM at $POLICY_BASE_URL"
    setsid env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}" \
        "$VLLM_BIN" serve "$MODEL_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --served-model-name "$POLICY_MODEL" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --data-parallel-size "$DATA_PARALLEL_SIZE" \
            --gpu-memory-utilization 0.6 \
            --max-model-len "$MAX_MODEL_LEN" \
            --dtype "$DTYPE" \
            >"$VLLM_LOG_FILE" 2>&1 &
    server_pid=$!
    server_started=1

    deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
    until is_server_ready; do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "vLLM exited before becoming ready. See $VLLM_LOG_FILE" >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for vLLM. See $VLLM_LOG_FILE" >&2
            exit 1
        fi
        sleep 2
    done
else
    echo "Policy server is unavailable at $POLICY_BASE_URL" >&2
    exit 1
fi

args=(
    "$SCRIPT_DIR/collect_baseline_rollouts.py"
    --output-dir "$OUTPUT_DIR"
    --output-path "$BASELINE_ROLLOUTS_JSONL"
    --num-tasks "$NUM_TASKS"
    --rollouts-per-task "$ROLLOUTS_PER_TASK"
    --task-batch-size "$TASK_BATCH_SIZE"
    --request-workers "$REQUEST_WORKERS"
    --max-steps "$MAX_STEPS"
    --seed "$SEED"
    --policy-base-url "$POLICY_BASE_URL"
    --policy-api-key "$POLICY_API_KEY"
    --policy-model "$POLICY_MODEL"
    --policy-temperature "$POLICY_TEMPERATURE"
    --policy-max-completion-tokens "$POLICY_MAX_COMPLETION_TOKENS"
    --policy-timeout "$POLICY_TIMEOUT"
    --policy-retries "$POLICY_RETRIES"
    --policy-retry-delay "$POLICY_RETRY_DELAY"
)
if [[ "$RESUME" == "true" ]]; then
    args+=(--resume)
fi
if [[ "$OVERWRITE" == "true" ]]; then
    args+=(--overwrite)
fi

echo "Collecting ${NUM_TASKS} x ${ROLLOUTS_PER_TASK} visual EZPoints baseline rollouts in waves of ${TASK_BATCH_SIZE} tasks."
PYTHONUNBUFFERED=1 python3 "${args[@]}"
