#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/local/user/models/qwen3_8b_awm_skill_sft_verl}"
BASE_PORT="${BASE_PORT:-8120}"
LOG_DIR="${LOG_DIR:-/workspace/idea_source/outputs/awm_qwen3_8b_eval/vllm_logs}"
VLLM="${VLLM:-/local/user/conda_envs/vllm_qwen_1/bin/vllm}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_ROOT/global_step_1600_vllm}"
STEPS=(100 300 500 700 900 1100 1300 1600)

mkdir -p "$LOG_DIR"
if [[ ! -x "$VLLM" ]]; then
  echo "vLLM executable not found: $VLLM" >&2
  exit 1
fi

for i in "${!STEPS[@]}"; do
  port=$((BASE_PORT + i))
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
    echo "ERROR: port $port is already in use" >&2
    exit 1
  fi
done

for i in "${!STEPS[@]}"; do
  step="${STEPS[$i]}"
  port=$((BASE_PORT + i))
  model_path="$MODEL_ROOT/global_step_${step}"
  model_name="qwen3-8b-step-${step}"
  if [[ ! -f "$model_path/model.safetensors" ]]; then
    echo "ERROR: missing model checkpoint $model_path" >&2
    exit 1
  fi
  echo "Starting step=$step gpu=$i port=$port model=$model_path"
  CUDA_VISIBLE_DEVICES="$i" "$VLLM" serve "$model_path" \
    --served-model-name "$model_name" \
    --tokenizer "$TOKENIZER_PATH" \
    --host 127.0.0.1 \
    --port "$port" \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    >"$LOG_DIR/step${step}.log" 2>&1 &
  echo $! >"$LOG_DIR/step${step}.pid"
done

for i in "${!STEPS[@]}"; do
  step="${STEPS[$i]}"
  port=$((BASE_PORT + i))
  for attempt in $(seq 1 180); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "READY step=$step gpu=$i port=$port"
      break
    fi
    if [[ "$attempt" == 180 ]]; then
      echo "ERROR: step=$step server on port $port did not become ready" >&2
      tail -80 "$LOG_DIR/step${step}.log" >&2 || true
      exit 1
    fi
    sleep 2
  done
done

echo "All checkpoint servers are ready."
wait
