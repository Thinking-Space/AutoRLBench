#!/usr/bin/env bash
# Whole-node AIME 2024 evaluation for DeepSeek-R1-Distill-Qwen-7B.
# Eight processes evaluate disjoint problems; no training is started.
# Prompt length is unlimited, thinking is enabled, and output is capped at 32K.
#SBATCH --export=ALL
set -euo pipefail

QAT_ROOT=/historical/home/QAT
PYTHON=${PYTHON:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python}
MODEL=${MODEL:-/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b}
DATA=${DATA:-/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet}
OUT=${OUT:-/historical/shared/qat/runs/aime/aime-8gpu-${SLURM_JOB_ID:-manual}}
NUM_SHARDS=${NUM_SHARDS:-8}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-42}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

[[ "$NUM_SHARDS" == 8 ]] || { echo 'NUM_SHARDS must be 8' >&2; exit 2; }
mkdir -p "$OUT/shards"
export QAT_ROOT PYTHON MODEL DATA OUT NUM_SHARDS NUM_SAMPLES SEED ATTN_IMPLEMENTATION
export PYTHONPATH="$QAT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

run_variant() {
  export KV_BITS="$1" EVAL_DTYPE="$2" REPORT_LABEL="$3"
  srun --ntasks=8 --ntasks-per-node=8 --gpus-per-task=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --exclusive bash -lc '
      exec "$PYTHON" "$QAT_ROOT/scripts/eval_aime.py" \
        --model "$MODEL" --data "$DATA" --profile aime_math \
        --kv-bits "$KV_BITS" --dtype "$EVAL_DTYPE" --thinking \
        --num-samples "$NUM_SAMPLES" --limit 30 --max-new-tokens 32768 \
        --attn-implementation "$ATTN_IMPLEMENTATION" --seed "$SEED" \
        --device cuda:0 --num-shards "$NUM_SHARDS" --shard-index "$SLURM_PROCID" \
        --progress --output "$OUT/shards/${REPORT_LABEL}-${SLURM_PROCID}.json"
    '
  shard_reports=("$OUT/shards/${REPORT_LABEL}-"*.json)
  [[ "${#shard_reports[@]}" -eq "$NUM_SHARDS" ]] || { echo "incomplete $REPORT_LABEL shards" >&2; exit 2; }
  "$PYTHON" "$QAT_ROOT/scripts/merge_aime_shards.py" \
    --inputs "${shard_reports[@]}" --output "$OUT/${REPORT_LABEL}.json"
}

run_variant 32 fp32 fp32
run_variant 2 bf16 kv2
run_variant 4 bf16 kv4

echo "AIME evaluation complete: $OUT"
