#!/usr/bin/env bash
# Merge one FSDP checkpoint with VeRL's official model merger, then evaluate it
# with native vLLM TurboQuant.
#
# This is the parameterised sibling of slurm_mopd_checkpoint_eval.sbatch, which
# hardcodes STEPS=(20..200) and --kv-cache-dtype turboquant_3bit_nc.  Neither
# matches the current runs: they train turboquant_k3v4_nc and save at whatever
# SAVE_FREQ says.  Evaluating a k3v4-trained checkpoint under a 3bit cache
# would answer a question nobody asked.
#
# Submit as an array indexed into STEPS, or with MERGE_ONLY=1 to validate just
# the merge half (CPU-bound, no GPU needed but the reservation is harmless):
#   export RUN_DIR=... STEPS=20,40,60,80,100,120
#   sbatch --array=0-5 --export=ALL \
# STEPS must be exported, not passed inside --export: Slurm splits that list
# on commas, so --export=...,STEPS=20,40,60 arrives as STEPS=20 and silently
# drops the rest.  That is what left j16889 with only task 0 alive.
#     scripts/slurm_qat_checkpoint_eval.sbatch
#
#SBATCH --job-name=qat-ckpt-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20d:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/historical/shared/qat/r/qat-eval/%x-%A_%a.out
#SBATCH --error=/historical/shared/qat/r/qat-eval/%x-%A_%a.err
set -euo pipefail
unset LD_PRELOAD

# MODEL_DIR evaluates an already-usable HF directory (e.g. the base model)
# and skips the merge entirely; RUN_DIR is then only used to place output.
if [[ -z "${MODEL_DIR:-}" ]]; then
  : "${RUN_DIR:?set RUN_DIR to the training run directory, or MODEL_DIR to an HF dir}"
fi
: "${RUN_DIR:=/historical/shared/qat/runs/qat-eval-standalone}"
PROFILE="${PROFILE:-deepseek_skywork}"
KV_DTYPE="${KV_DTYPE:-turboquant_k3v4_nc}"   # or "auto" for the unquantized arm
STEPS="${STEPS:-20,40,60,80,100,120}"
LIMIT="${LIMIT:-}"
# vLLM queues past this rather than OOMing, so a high value is safe:
# it schedules as many as the KV budget actually allows.
N="${N:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

case "$PROFILE" in
  deepseek_skywork)
    BENCHMARK=aime_math
    DATA=/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet
    # Measured on this dataset: max_prompt_tokens=427, and sampling caps
    # max_tokens at 32768.  131072 was 4x oversized and the wasted KV budget
    # is what forced max_num_seqs down to 4 -- which made the whole eval
    # single-stream (55.5 tok/s/seq, 30 sequential waves for 120 samples).
    MAX_MODEL_LEN=33792
    ;;
  minerva_math)
    BENCHMARK=minerva_math
    DATA=/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/minerva_math/test.jsonl
    # Longest prompt is 423 tokens; this preserves the same 32K output cap.
    MAX_MODEL_LEN=33792
    ;;
  olympiad_bench)
    BENCHMARK=olympiad_bench
    DATA=/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/olympiadbench/test.jsonl
    # Longest prompt is 1284 tokens, so AIME's 33792 would silently reduce
    # the requested 32768-token output cap on the longest questions.
    MAX_MODEL_LEN=34816
    ;;
  qwen25_longrlvr)
    BENCHMARK=longbench_v2_medium
    DATA=
    MAX_MODEL_LEN=1010000
    ;;
  *) echo "unsupported PROFILE=$PROFILE" >&2; exit 2 ;;
esac

IFS=',' read -r -a STEP_LIST <<< "$STEPS"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK_ID >= 0 && TASK_ID < ${#STEP_LIST[@]} )) || { echo "invalid array index $TASK_ID for STEPS=$STEPS" >&2; exit 2; }
STEP="${STEP_LIST[$TASK_ID]}"

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
# Vendored verl, not the /shared reference copy: the checkpoints were written
# by this tree's FSDP config and the merger reads fsdp_config.json from it.
# turboquant-train-overlay is not optional: it is the only place msgpack
# lives, and mopd-pylibs' ray shadows any other copy on sys.path, so
# omitting it makes "import verl" -> "import ray" fail outright.
export PYTHONPATH="$ROOT:$ROOT/vendor/verl:/historical/shared/qat/tmp/turboquant-train-overlay:/historical/shared/qat/envs/mopd-pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ENV/bin:$PATH"
# Triton JIT-compiles cuda_utils.c at engine init and needs Python.h.
# The nodes have no python3-devel; the vendored headers are the only copy.
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PYINC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

CHECKPOINT="$RUN_DIR/checkpoints/global_step_${STEP}/actor"
MERGED="$RUN_DIR/merged/global_step_${STEP}"
OUTPUT="$RUN_DIR/eval/${KV_DTYPE}_step${STEP}.json"
mkdir -p /historical/shared/qat/r/qat-eval "$RUN_DIR/eval"

export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

if [[ -n "${MODEL_DIR:-}" ]]; then
  MERGED="$MODEL_DIR"
  OUTPUT="$RUN_DIR/eval/${TAG:-model}_${KV_DTYPE}.json"
  echo "[qat-eval] MODEL_DIR set; skipping merge, evaluating $MERGED"
else
[[ -d "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }

if [[ ! -f "$MERGED/config.json" || ! -f "$MERGED/model.safetensors.index.json" ]]; then
  echo "[qat-eval] merging $CHECKPOINT -> $MERGED"
  rm -rf "$MERGED.tmp-${SLURM_JOB_ID}"
  "$ENV/bin/python" -m verl.model_merger merge \
    --backend fsdp --local_dir "$CHECKPOINT" \
    --target_dir "$MERGED.tmp-${SLURM_JOB_ID}" \
    --trust-remote-code --use_cpu_initialization
  # The merger writes weights only; the tokenizer lives beside the shards.
  cp -n "$CHECKPOINT/huggingface/"* "$MERGED.tmp-${SLURM_JOB_ID}/" 2>/dev/null || true
  mv "$MERGED.tmp-${SLURM_JOB_ID}" "$MERGED"
  echo "[qat-eval] merge complete"
else
  echo "[qat-eval] reusing existing merge at $MERGED"
fi
fi

ls -la "$MERGED"

if [[ "${MERGE_ONLY:-0}" == "1" ]]; then
  echo "[qat-eval] MERGE_ONLY=1; stopping before evaluation"
  exit 0
fi

EVAL_ARGS=(--benchmark "$BENCHMARK" --model "$MERGED")
[[ -n "$DATA"  ]] && EVAL_ARGS+=(--data "$DATA")
[[ -n "$LIMIT" ]] && EVAL_ARGS+=(--limit "$LIMIT")

echo "[qat-eval] evaluating step=$STEP kv_cache_dtype=$KV_DTYPE -> $OUTPUT"
exec "$ENV/bin/python" "$ROOT/scripts/eval_vllm_turboquant.py" \
  "${EVAL_ARGS[@]}" \
  --kv-cache-dtype "$KV_DTYPE" --output "$OUTPUT" \
  --n "$N" --tensor-parallel-size 2 --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 32768 --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 --seed 42 --progress
