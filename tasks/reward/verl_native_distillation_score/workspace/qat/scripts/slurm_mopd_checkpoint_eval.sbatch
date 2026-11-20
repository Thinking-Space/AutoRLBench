#!/usr/bin/env bash
# Evaluate every saved OPD checkpoint with the same benchmark contract as the
# base-model TurboQuant runs.  This is an asynchronous post-training array:
# each task merges one FSDP checkpoint with VeRL's official model merger, then
# runs native vLLM TurboQuant 3bit evaluation.  No training logic is duplicated.
#
# Submit after the corresponding main job is known, for example:
#   sbatch --dependency=afterok:7024 --array=0-9 \
#     --export=ALL,PROFILE=deepseek_skywork,RUN_DIR=/historical/shared/qat/runs/mopd/deepseek_skywork/kv32/main200 \
#     scripts/slurm_mopd_checkpoint_eval.sbatch
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20d:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --output=/historical/shared/qat/r/mopd-eval-logs/%x-%A_%a.out
#SBATCH --error=/historical/shared/qat/r/mopd-eval-logs/%x-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail
: "${PROFILE:?set PROFILE=deepseek_skywork or qwen25_longrlvr}"
: "${RUN_DIR:?set RUN_DIR to the completed VeRL run directory}"

case "$PROFILE" in
  deepseek_skywork)
    BENCHMARK=aime_math
    DATA=/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet
    MAX_MODEL_LEN=131072
    ;;
  qwen25_longrlvr)
    BENCHMARK=longbench_v2_medium
    # The local data.json is not present on this machine; the evaluator's
    # pinned HF loader is the authoritative LongBench v2 source.
    DATA=
    MAX_MODEL_LEN=1010000
    ;;
  *) echo "unsupported PROFILE=$PROFILE" >&2; exit 2 ;;
esac

STEPS=(20 40 60 80 100 120 140 160 180 200)
TASK_ID="${SLURM_ARRAY_TASK_ID:?this file must be submitted as an array}"
(( TASK_ID >= 0 && TASK_ID < ${#STEPS[@]} )) || { echo "invalid array index $TASK_ID" >&2; exit 2; }
STEP="${STEPS[$TASK_ID]}"

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
MERGE_ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PYTHONPATH="$ROOT:/historical/shared/qat/tmp/reference/Open-MOPD/training/verl:/historical/shared/qat/envs/mopd-pylibs${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="$RUN_DIR/checkpoints/global_step_${STEP}/actor"
MERGED="$RUN_DIR/merged/global_step_${STEP}"
OUTPUT="$RUN_DIR/eval/turboquant_3bit_nc_step${STEP}.json"
mkdir -p /historical/shared/qat/r/mopd-eval-logs "$RUN_DIR/eval"
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

[[ -d "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }
if [[ ! -f "$MERGED/config.json" || ! -f "$MERGED/model.safetensors.index.json" ]]; then
  rm -rf "$MERGED.tmp-${SLURM_JOB_ID}"
  "$MERGE_ENV/bin/python" -m verl.model_merger merge \
    --backend fsdp --local_dir "$CHECKPOINT" \
    --target_dir "$MERGED.tmp-${SLURM_JOB_ID}" \
    --trust-remote-code --use_cpu_initialization
  mv "$MERGED.tmp-${SLURM_JOB_ID}" "$MERGED"
fi

EVAL_ARGS=(--benchmark "$BENCHMARK" --model "$MERGED")
if [[ -n "$DATA" ]]; then
  EVAL_ARGS+=(--data "$DATA")
fi
exec "$ENV/bin/python" "$ROOT/scripts/eval_vllm_turboquant.py" \
  "${EVAL_ARGS[@]}" \
  --kv-cache-dtype turboquant_3bit_nc --output "$OUTPUT" \
  --tensor-parallel-size 2 --max-num-seqs 4 \
  --max-num-batched-tokens 32768 --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 --seed 42 --progress
