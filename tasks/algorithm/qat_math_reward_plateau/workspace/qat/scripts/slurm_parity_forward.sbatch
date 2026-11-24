#!/bin/bash
#SBATCH --job-name=qat-parity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/historical/shared/qat/r/qat-parity/parity-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-parity/parity-%j.err
#
# Forward-level parity: does the training fake quantizer reproduce what vLLM
# actually computes?  See scripts/parity_forward_logprobs.py for the design.
#
# Each phase runs as its own process so the 7B model is never resident twice
# and a vLLM engine never has to be torn down mid-process.
set -euo pipefail

# The agent shell exports LD_PRELOAD=libproxychains4.so; with --export=ALL it is
# injected into every job process.  Strip it (it hung Ray actor startup once
# already and there is no reason to carry it here).
unset LD_PRELOAD

ROOT=/historical/home/QAT
VENV_BIN="${MOPD_VENV_BIN:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin}"
export PATH="$VENV_BIN:$PATH"
PY="$VENV_BIN/python"

KV_SCHEME="${KV_SCHEME:-turboquant_k3v4_nc}"
# The script's DEFAULT_MODEL is the 7B.  A 1.5B run that does not set this
# silently measures parity on a model it never trains, so make it explicit.
MODEL="${MODEL:-/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b}"
PROMPT_PARQUET="${PROMPT_PARQUET:-}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
# Real responses on this profile average ~6000 tokens, so a short
# completion would under-test how quantization error compounds as the
# cache grows.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_ROW="${PROMPT_ROW:-0}"
# 0.0 = greedy.  Set 1.0 to reproduce the training rollout regime: sampled
# tokens come from the tail of the distribution, where two quantizers
# disagree far more than they do on the argmax token.
TEMPERATURE="${TEMPERATURE:-0.0}"

OUT="/historical/shared/qat/runs/qat-parity/$(basename "$MODEL")-${KV_SCHEME}-r${PROMPT_ROW}-t${TEMPERATURE}-j${SLURM_JOB_ID}"
mkdir -p "$OUT" /historical/shared/qat/r/qat-parity

export PYTHONPATH="$ROOT"
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM refuses to ship a Python callable to a worker process unless this
# is set, which blocks the collective_rpc path qat_kv.native_tq_verify
# uses to inspect attention layers built in the EngineCore subprocess.
# Safe here: the only callable sent is our own, in our own job.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
# flashinfer/Triton JIT need Python.h, which the compute nodes lack.
INCLUDE=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$INCLUDE" C_INCLUDE_PATH="$INCLUDE" CPLUS_INCLUDE_PATH="$INCLUDE"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

cd "$ROOT"
COMMON=(--model "$MODEL"
        --max-prompt-tokens "$MAX_PROMPT_TOKENS"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --prompt-row "$PROMPT_ROW"
        --temperature "$TEMPERATURE")
[[ -n "$PROMPT_PARQUET" ]] && COMMON+=(--prompt-parquet "$PROMPT_PARQUET")
# The hf phase loads the model too, and defaults to the 7B just like the
# vllm phase does.
HFCOMMON=(--model "$MODEL")

echo "[parity] === 1/5 vLLM, quantized ($KV_SCHEME) ==="
$PY scripts/parity_forward_logprobs.py --phase vllm --kv-scheme "$KV_SCHEME" \
  "${COMMON[@]}" --out "$OUT/tq_vllm.json"

echo "[parity] === 2/5 HF, teacher-forced on the quantized run's tokens ==="
$PY scripts/parity_forward_logprobs.py --phase hf "${HFCOMMON[@]}" --tokens "$OUT/tq_vllm.json" \
  --out "$OUT/tq_hf.json"

echo "[parity] === 3/5 vLLM, full-precision control ==="
$PY scripts/parity_forward_logprobs.py --phase vllm --kv-scheme none \
  "${COMMON[@]}" --out "$OUT/fp_vllm.json"

echo "[parity] === 4/5 HF, teacher-forced on the control run's tokens ==="
$PY scripts/parity_forward_logprobs.py --phase hf "${HFCOMMON[@]}" --tokens "$OUT/fp_vllm.json" \
  --out "$OUT/fp_hf.json"

echo "[parity] === 5/5 report ==="
$PY scripts/parity_forward_logprobs.py --phase report \
  --quantized "$OUT/tq_hf.json" --control "$OUT/fp_hf.json" \
  --out "$OUT/report.json"

echo "[parity] done: $OUT/report.json"
