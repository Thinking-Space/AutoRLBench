#!/bin/bash
#SBATCH --job-name=qat-kl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/historical/shared/qat/logs/kl-probe-%A_%a.out
#SBATCH --error=/historical/shared/qat/logs/kl-probe-%A_%a.err

set -euo pipefail
unset LD_PRELOAD

ROOT=/historical/home/QAT
PY=/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python
BASE=/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl
SAMPLES=/historical/shared/qat/runs/qat-kl-probe-1p5b/balanced-30train-30test.json
OUT="${OUT_DIR:-/historical/shared/qat/runs/qat-kl-probe-1p5b/results}"

OPD=/historical/shared/qat/runs/qat-train/deepseek_skywork/turboquant_k3v4_nc-ba0.0-g8-b64-lr1.5e-6-k512-union-opd-only-pl512-j8162/merged
NMSE=/historical/shared/qat/runs/qat-train/deepseek_skywork/turboquant_k3v4_nc-ba1.0-g4-b64-lr1.5e-6-k512-union-nmse-only-g4-pl512-j8170/merged
OFF=/historical/shared/qat/runs/qat-train/deepseek_skywork/turboquant_k3v4_nc-ba0.0-g4-b64-lr1.5e-6-k512-union-llmqat-offline-revkl-pl512-j8415/merged

case "$SLURM_ARRAY_TASK_ID" in
  0) LABEL=opd-s60;     CANDIDATE="$OPD/global_step_60" ;;
  1) LABEL=opd-s150;    CANDIDATE="$OPD/global_step_150" ;;
  2) LABEL=nmse-s60;    CANDIDATE="$NMSE/global_step_60" ;;
  3) LABEL=nmse-s150;   CANDIDATE="$NMSE/global_step_150" ;;
  4) LABEL=offdist-s60; CANDIDATE="$OFF/global_step_60" ;;
  5) LABEL=offdist-s150; CANDIDATE="$OFF/global_step_150" ;;
  *) echo "invalid array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

mkdir -p "$OUT" /historical/shared/qat/logs
export PYTHONPATH="$ROOT"
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"

cd "$ROOT"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
exec "$PY" scripts/kl_checkpoint_vs_base.py \
  --base "$BASE" \
  --candidate "$CANDIDATE" \
  --label "$LABEL" \
  --samples "$SAMPLES" \
  --kv-scheme turboquant_k3v4_nc \
  --logit-chunk 32 \
  "${LIMIT_ARGS[@]}" \
  --output "$OUT/$LABEL.json"
