#!/usr/bin/env bash
# Unquantized, no-CoT-prompt LongBench v2 medium baseline for KIVI comparison.
#SBATCH --export=ALL
set -euo pipefail
QAT_ROOT=/historical/home/QAT
PYTHON=${PYTHON:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python}
MODEL=${MODEL:-/historical/shared/qat/models/base/qwen2.5-7b-instruct-1m}
OUT=${OUT:-/historical/shared/qat/runs/longbench_v2/bf16-nothinking-${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT/shards"
export QAT_ROOT PYTHON MODEL OUT
export PYTHONPATH="$QAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
srun --ntasks=8 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --exclusive bash -lc '
  exec "$PYTHON" "$QAT_ROOT/scripts/eval_longbench_v2.py" --model "$MODEL" \
    --subset medium --difficulty all --expected-rows 215 --limit 215 --kv-bits 32 --no-thinking \
    --temperature 0.1 --top-p 1.0 --top-k -1 --max-new-tokens 4096 --device cuda:0 --dtype bf16 \
    --attn-implementation sdpa --seed 42 --num-shards 8 --shard-index "$SLURM_PROCID" \
    --progress --output "$OUT/shards/bf16-${SLURM_PROCID}.json"
'
shards=("$OUT/shards/bf16-"*.json)
[[ "${#shards[@]}" -eq 8 ]] || { echo "incomplete baseline shards" >&2; exit 2; }
"$PYTHON" "$QAT_ROOT/scripts/merge_longbench_shards.py" --inputs "${shards[@]}" --output "$OUT/bf16.json"
