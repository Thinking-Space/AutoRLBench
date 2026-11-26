#!/usr/bin/env bash
# Run several AIME evaluations concurrently on one exclusive node.
#
# Two problems this solves that the per-config array job could not:
#
#  1. The cluster is shared and GPU memory is not isolated between jobs.  A
#     2-GPU eval landing next to another user's process dies at engine init
#     ("Free memory on device cuda:1 (25.35/267.69 GiB) ... less than desired
#     GPU memory utilization").  --exclusive removes the co-tenant entirely.
#  2. tensor_parallel_size=2 buys nothing for a 7B model on a 267 GiB card and
#     costs TP communication.  TP=1 lets one config own one GPU, so an
#     exclusive node runs 8 configs at once instead of 4.
#
# SPECS is a ';'-separated list of  label|source|step|kv_dtype  where source is
# either an HF model directory or a training RUN_DIR (then step selects the
# checkpoint and it is merged first).
#
#   sbatch --export=ALL,SPECS="base-fp|/path/model||auto;s20|/path/run|20|turboquant_k3v4_nc" \
#     scripts/slurm_qat_eval_pack.sbatch
#
#SBATCH --job-name=qat-eval-pack
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:nvidia_l20d:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --output=/historical/shared/qat/r/qat-eval/pack-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-eval/pack-%j.err
set -euo pipefail
unset LD_PRELOAD

: "${SPECS:?set SPECS}"
OUT_ROOT="${OUT_ROOT:-/historical/shared/qat/runs/qat-eval-pack}"
# Samples per problem.  The caller sets this so every benchmark has at least
# 400 total samples (for example AMC23: 40 x 16; OlympiadBench: 675 x 1).
N="${N:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PROFILE="${PROFILE:-aime_2024}"
case "$PROFILE" in
  aime_2024|aime_2025)
    BENCHMARK=aime_math
    DATA=/historical/shared/qat/data/raw/${PROFILE}/data/train-00000-of-00001.parquet
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-33792}"
    ;;
  amc23)
    BENCHMARK=amc23
    DATA=/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/amc23/test.jsonl
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-33792}"
    ;;
  olympiad_bench)
    BENCHMARK=olympiad_bench
    DATA=/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval/data/olympiadbench/test.jsonl
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
    ;;
  *) echo "unsupported PROFILE=$PROFILE" >&2; exit 2 ;;
esac
[[ -f "$DATA" ]] || { echo "missing dataset: $DATA" >&2; exit 2; }

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PATH="$ENV/bin:$PATH"
# msgpack lives only in the train overlay, and mopd-pylibs' ray shadows any
# other copy, so "import verl" (needed by the merger) fails without it.
export PYTHONPATH="$ROOT:$ROOT/vendor/verl:/historical/shared/qat/tmp/turboquant-train-overlay:/historical/shared/qat/envs/mopd-pylibs"
# Triton JIT-compiles cuda_utils.c at engine init; the nodes have no python3-devel.
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC" C_INCLUDE_PATH="$PYINC" CPLUS_INCLUDE_PATH="$PYINC"
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"

mkdir -p /historical/shared/qat/r/qat-eval "$OUT_ROOT/eval" "$OUT_ROOT/logs"

IFS=';' read -r -a SPEC_LIST <<< "$SPECS"
echo "[pack] ${#SPEC_LIST[@]} configs, profile=$PROFILE benchmark=$BENCHMARK n=$N max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"

# --- phase 1: merges (CPU bound, sequential so they do not thrash memory) ---
declare -a MODELS LABELS DTYPES
for i in "${!SPEC_LIST[@]}"; do
  IFS='|' read -r label source step kv <<< "${SPEC_LIST[$i]}"
  if [[ -n "$step" ]]; then
    merged="$source/merged/global_step_${step}"
    ckpt="$source/checkpoints/global_step_${step}/actor"
    if [[ ! -f "$merged/config.json" ]]; then
      echo "[pack] merging $ckpt"
      rm -rf "$merged.tmp-$$"
      "$ENV/bin/python" -m verl.model_merger merge --backend fsdp \
        --local_dir "$ckpt" --target_dir "$merged.tmp-$$" \
        --trust-remote-code --use_cpu_initialization
      cp -n "$ckpt/huggingface/"* "$merged.tmp-$$/" 2>/dev/null || true
      mv "$merged.tmp-$$" "$merged"
    else
      echo "[pack] reusing merge $merged"
    fi
    MODELS[$i]="$merged"
  else
    MODELS[$i]="$source"
  fi
  LABELS[$i]="$label"; DTYPES[$i]="$kv"
done

# --- phase 2: one GPU per config, all at once ---
pids=()
for i in "${!SPEC_LIST[@]}"; do
  gpu=$(( i % 8 ))
  out="$OUT_ROOT/eval/${LABELS[$i]}.json"
  log="$OUT_ROOT/logs/${LABELS[$i]}.log"
  echo "[pack] gpu$gpu <- ${LABELS[$i]} (${DTYPES[$i]}) -> $out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TRITON_CACHE_DIR="/tmp/pack-triton-${SLURM_JOB_ID}-$i"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/pack-inductor-${SLURM_JOB_ID}-$i"
    export VLLM_CACHE_ROOT="/tmp/pack-vllm-${SLURM_JOB_ID}-$i"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
    "$ENV/bin/python" "$ROOT/scripts/eval_vllm_turboquant.py" \
      --benchmark "$BENCHMARK" --model "${MODELS[$i]}" --data "$DATA" \
      --kv-cache-dtype "${DTYPES[$i]}" --output "$out" \
      --n "$N" --tensor-parallel-size 1 --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens 32768 --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.90 --seed 42 --progress
  ) >"$log" 2>&1 &
  pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[pack] OK   ${LABELS[$i]}"
  else echo "[pack] FAIL ${LABELS[$i]} (see $OUT_ROOT/logs/${LABELS[$i]}.log)"; rc=1; fi
done
echo "[pack] done rc=$rc"
exit $rc
