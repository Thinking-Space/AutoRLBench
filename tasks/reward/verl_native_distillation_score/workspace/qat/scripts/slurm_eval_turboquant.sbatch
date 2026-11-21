#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20d:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --output=/historical/shared/qat/r/turboquant-evals/logs/%x-%j.out
#SBATCH --error=/historical/shared/qat/r/turboquant-evals/logs/%x-%j.err
set -euo pipefail

: "${BENCHMARK:?set BENCHMARK=aime_math or longbench_v2_medium}"
: "${METHOD:?set METHOD=turboquant_3bit_nc or turboquant_k3v4_nc}"
: "${MODEL:?set MODEL to a local checkpoint}"
: "${OUTPUT:?set OUTPUT to the JSON report path}"

ENV_DIR=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PATH="$ENV_DIR/bin:$PATH"
export CPATH=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
# Triton/Inductor compilation must not write temporary kernels to the home
# BeeGFS mount.  A transient Remote I/O error there can kill an otherwise
# completed multi-hour evaluation.  Keep per-job caches on the allocated
# node; they are disposable and cannot collide across concurrent jobs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
mkdir -p "$(dirname "$OUTPUT")" /historical/shared/qat/r/turboquant-evals/logs

ARGS=(
  --benchmark "$BENCHMARK"
  --model "$MODEL"
  --kv-cache-dtype "$METHOD"
  --output "$OUTPUT"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
  --max-num-seqs "${MAX_NUM_SEQS:-2}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --max-model-len "${MAX_MODEL_LEN:?set MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --n "${N:-4}"
  --seed "${SEED:-42}"
  --progress
)
if [[ -n "${DATA:-}" ]]; then
  ARGS+=(--data "$DATA")
fi
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

exec "$ENV_DIR/bin/python" /historical/home/QAT/scripts/eval_vllm_turboquant.py "${ARGS[@]}"
