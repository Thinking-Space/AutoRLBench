#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/local/user/Qwen3.8-27B}"
MODEL_NAME="${MODEL_NAME:-qwen38-27b}"
PORT="${PORT:-8200}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-/workspace/idea_source/outputs/awm_vllm_qwen38}"

mkdir -p "$LOG_DIR"
if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q .; then
  echo "ERROR: port $PORT is already in use" >&2
  exit 1
fi

echo "Launching Qwen3.8-27B on GPUs $GPU_DEVICES, port $PORT"
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --trust-remote-code \
  2>&1 | tee "$LOG_DIR/server.log"
