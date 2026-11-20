#!/usr/bin/env bash
# DeepSeek-R1-Distill-Qwen-7B AIME 2024 KV4 evaluation.
# Runs only after the preceding KV2 job succeeds.
# Prompt length is unlimited, Thinking is enabled, output is capped at 32K.
#SBATCH --export=ALL
set -euo pipefail

QAT_ROOT=/historical/home/QAT
PYTHON=${PYTHON:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python}
MODEL=${MODEL:-/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b}
DATA=${DATA:-/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet}
OUT=${OUT:-/historical/shared/qat/runs/aime/kv4-${SLURM_JOB_ID:-manual}}
NUM_SHARDS=8
NUM_SAMPLES=${NUM_SAMPLES:-4}
WORK_OUT=${SLURM_TMPDIR:-/tmp}/aime-kv4-${SLURM_JOB_ID:-manual}

rm -rf "$WORK_OUT"
mkdir -p "$OUT" "$WORK_OUT/shards"
export QAT_ROOT PYTHON MODEL DATA OUT WORK_OUT NUM_SHARDS NUM_SAMPLES
export PYTHONPATH="$QAT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1

srun --ntasks=8 --ntasks-per-node=8 --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --exclusive bash -lc '
    exec "$PYTHON" "$QAT_ROOT/scripts/eval_aime.py" \
      --model "$MODEL" --data "$DATA" --profile aime_math \
      --kv-bits 4 --dtype bf16 --thinking \
      --num-samples "$NUM_SAMPLES" --limit 30 --max-new-tokens 32768 \
      --attn-implementation sdpa --seed 42 --device cuda:0 \
      --num-shards 8 --shard-index "$SLURM_PROCID" \
      --progress --output "$WORK_OUT/shards/kv4-${SLURM_PROCID}.json"
  '

shards=("$WORK_OUT/shards/kv4-"*.json)
[[ "${#shards[@]}" -eq 8 ]] || { echo "incomplete KV4 AIME shards" >&2; exit 2; }
"$PYTHON" "$QAT_ROOT/scripts/merge_aime_shards.py" \
  --inputs "${shards[@]}" --output "$WORK_OUT/kv4.json"
cp "$WORK_OUT/kv4.json" "$OUT/kv4.json"
echo "AIME KV4 evaluation complete: $OUT/kv4.json"
