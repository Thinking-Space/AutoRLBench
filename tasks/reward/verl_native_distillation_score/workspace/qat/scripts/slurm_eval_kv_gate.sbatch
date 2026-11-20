#!/usr/bin/env bash
# Independent FP32 -> KV2 -> conditional-KV4 diagnostic.  DeepSeek uses AIME
# 2024; Qwen uses only LongBench v2 ``length=medium`` (all difficulties).  This job is
# deliberately separate from the live MOPD trainer: it requests its own
# Slurm allocation and never writes to a trainer Ray/checkpoint directory.
#
# DeepSeek/Skywork (the default):
#   sbatch --partition=gpu --nodelist=lj-gpu2 --gres=gpu:1 \
#     --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 --mem=128G \
#     --time=08:00:00 --job-name=qat-kv-gate \
#     --output=/historical/shared/qat/runs/aime/kv-gate-%j.log \
#     --error=/historical/shared/qat/runs/aime/kv-gate-%j.log \
#     scripts/slurm_eval_kv_gate.sbatch
#
# Qwen/LongRLVR uses LongBench v2 medium without editing this script:
#   sbatch --export=ALL,PROFILE=qwen25_longrlvr ... \
#     scripts/slurm_eval_kv_gate.sbatch
#
# LongBench v2 uses eight independent evaluator processes on an eight-GPU
# node.  Each process evaluates a disjoint shard; reports are merged before
# the FP/KV2/KV4 comparison.  AIME keeps the single-GPU path below.

#SBATCH --export=ALL
set -euo pipefail

QAT_ROOT=/historical/home/QAT
PYTHON=${PYTHON:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python}
PROFILE=${PROFILE:-deepseek_skywork}
DATA=${DATA:-}
OUT=${OUT:-}
# Slurm expands ``%j`` in --output/--error templates, but not inside values
# supplied through ``--export``.  Normalize it here so an explicit
# ``OUT=...-%j`` cannot make two jobs share a literal directory.
LIMIT=${LIMIT:-}
NUM_SAMPLES=${NUM_SAMPLES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
SEED=${SEED:-42}
DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
ALWAYS_EVALUATE_KV4=${ALWAYS_EVALUATE_KV4:-0}
NUM_SHARDS=${NUM_SHARDS:-8}

case "$PROFILE" in
  deepseek_skywork)
    MODEL=${MODEL:-/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b}
    TRAINING_DATA=${TRAINING_DATA:-/historical/shared/qat/data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet}
    DATA=${DATA:-/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet}
    EVAL_PROFILE=${EVAL_PROFILE:-aime_math}
    LIMIT=${LIMIT:-30}
    NUM_SAMPLES=${NUM_SAMPLES:-4}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
    ;;
  qwen25_longrlvr)
    MODEL=${MODEL:-/historical/shared/qat/models/base/qwen2.5-7b-instruct-1m}
    TRAINING_DATA=${TRAINING_DATA:-/historical/shared/qat/data/processed/longrlvr_qwen25_1m_prompt65536.parquet}
    DATA=${DATA:-/historical/shared/qat/data/raw/longbench-v2/data.json}
    EVAL_PROFILE=${EVAL_PROFILE:-longbench_v2_medium}
    LIMIT=${LIMIT:-215}
    NUM_SAMPLES=${NUM_SAMPLES:-1}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
    ;;
  *)
    echo "unsupported PROFILE=$PROFILE (expected deepseek_skywork or qwen25_longrlvr)" >&2
    exit 2
    ;;
esac

# Keep the two evaluator artifact trees separate.  An explicit OUT still wins.
if [[ -z "${OUT:-}" ]]; then
  if [[ "$PROFILE" == "qwen25_longrlvr" ]]; then
    OUT="/historical/shared/qat/runs/longbench_v2/kv-gate-${PROFILE}-${SLURM_JOB_ID:-manual}"
  else
    OUT="/historical/shared/qat/runs/aime/kv-gate-${PROFILE}-${SLURM_JOB_ID:-manual}"
  fi
fi
if [[ "$OUT" == *%j* ]]; then
  OUT="${OUT//%j/${SLURM_JOB_ID:-manual}}"
fi

export QAT_ROOT PYTHON PROFILE MODEL DATA TRAINING_DATA OUT EVAL_PROFILE
export LIMIT NUM_SAMPLES MAX_NEW_TOKENS SEED DEVICE DTYPE ATTN_IMPLEMENTATION ALWAYS_EVALUATE_KV4
export NUM_SHARDS

mkdir -p "$OUT"
export PYTHONPATH="$QAT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

EXTRA_ARGS=()
if [[ "$ALWAYS_EVALUATE_KV4" == "1" ]]; then
  EXTRA_ARGS+=(--always-evaluate-kv4)
fi

if [[ "$PROFILE" == "qwen25_longrlvr" ]]; then
  if [[ "$NUM_SHARDS" != "8" ]]; then
    echo "LongBench whole-node evaluation requires NUM_SHARDS=8" >&2
    exit 2
  fi
  mkdir -p "$OUT/shards"

  run_longbench_variant() {
    export KV_BITS="$1"
    export EVAL_DTYPE="$2"
    export REPORT_LABEL="$3"
    srun --ntasks="$NUM_SHARDS" --ntasks-per-node="$NUM_SHARDS" \
      --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --exclusive \
      bash -lc '
        set -euo pipefail
        data_args=()
        if [[ -f "$DATA" ]]; then data_args=(--data "$DATA"); fi
        exec "$PYTHON" "$QAT_ROOT/scripts/eval_longbench_v2.py" \
          --model "$MODEL" "${data_args[@]}" \
          --subset medium --difficulty all --expected-rows 215 --limit 215 \
          --kv-bits "$KV_BITS" --preserve-projection-bias --no-thinking \
          --temperature 0.1 --top-p 1.0 --top-k -1 \
          --max-new-tokens "$MAX_NEW_TOKENS" --device cuda:0 \
          --dtype "$EVAL_DTYPE" --attn-implementation "$ATTN_IMPLEMENTATION" \
          --seed "$SEED" --num-shards "$NUM_SHARDS" --shard-index "$SLURM_PROCID" \
          --progress --output "$OUT/shards/${REPORT_LABEL}-${SLURM_PROCID}.json"
      '

    shard_reports=("$OUT/shards/${REPORT_LABEL}-"*.json)
    if [[ "${#shard_reports[@]}" -ne "$NUM_SHARDS" ]]; then
      echo "expected $NUM_SHARDS $REPORT_LABEL shard reports, found ${#shard_reports[@]}" >&2
      exit 2
    fi
    "$PYTHON" "$QAT_ROOT/scripts/merge_longbench_shards.py" \
      --inputs "${shard_reports[@]}" --output "$OUT/${REPORT_LABEL}.json"
  }

  # The unquantized reference uses BF16 so 100k+ token prompts retain the
  # efficient attention path.  KV2 and KV4 use the same compute dtype.
  run_longbench_variant 32 bf16 bf16
  run_longbench_variant 2 bf16 kv2
  run_longbench_variant 4 bf16 kv4

  "$PYTHON" "$QAT_ROOT/scripts/eval_kv_gate.py" \
    --profile "$PROFILE" --model "$MODEL" --training-data "$TRAINING_DATA" \
    --evaluator-profile "$EVAL_PROFILE" --output-dir "$OUT" \
    --fp-report "$OUT/bf16.json" --kv2-report "$OUT/kv2.json" \
    --kv4-report "$OUT/kv4.json" --summary "$OUT/gate_selection.json" \
    --limit "$LIMIT" --num-samples "$NUM_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" --device cuda:0 --dtype bf16 \
    --attn-implementation "$ATTN_IMPLEMENTATION" --seed "$SEED" --json
  exit 0
fi

srun --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --exclusive \
  "$PYTHON" "$QAT_ROOT/scripts/eval_kv_gate.py" \
    --profile "$PROFILE" \
    --model "$MODEL" \
    --data "$DATA" \
    --training-data "$TRAINING_DATA" \
    --evaluator-profile "$EVAL_PROFILE" \
    --output-dir "$OUT" \
    --python "$PYTHON" \
    --limit "$LIMIT" \
    --num-samples "$NUM_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPLEMENTATION" \
    --seed "$SEED" \
    --run \
    --json \
    "${EXTRA_ARGS[@]}"
