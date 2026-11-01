#!/usr/bin/env bash
set -euo pipefail

# Launch one OpenAI-compatible Qwen3.5-9B vLLM server per GPU.
# This script is intended to be run inside persistent tmux panes.

MODEL_PATH="${MODEL_PATH:-/local/user/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-qwen35-9b}"
BASE_PORT="${BASE_PORT:-8020}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LOG_DIR="${LOG_DIR:-/workspace/idea_source/outputs/awm_vllm_qwen35}"

mkdir -p "$LOG_DIR"

echo "Qwen3.5-9B server configuration"
echo "  model=$MODEL_PATH"
echo "  ports=${BASE_PORT}-$((BASE_PORT + 7))"
echo "  max_num_seqs=$MAX_NUM_SEQS"
echo "  gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "  max_model_len=$MAX_MODEL_LEN"
echo "  logs=$LOG_DIR"

for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
    echo "ERROR: port $port is already in use" >&2
    exit 1
  fi
done

for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  echo "Starting GPU $gpu on port $port"
  CUDA_VISIBLE_DEVICES="$gpu" \
    vllm serve "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$port" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --trust-remote-code \
      >"$LOG_DIR/gpu${gpu}.log" 2>&1 &
  echo $! >"$LOG_DIR/gpu${gpu}.pid"
done

echo "Waiting for all OpenAI endpoints..."
for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  for attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "READY gpu=$gpu port=$port"
      break
    fi
    if [[ "$attempt" == 180 ]]; then
      echo "ERROR: server on port $port did not become ready" >&2
      tail -80 "$LOG_DIR/gpu${gpu}.log" >&2 || true
      exit 1
    fi
    sleep 2
  done
done

echo "All eight Qwen3.5-9B servers are ready."
echo "This launcher stays alive so the tmux window remains usable."
wait
