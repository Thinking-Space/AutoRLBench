#!/usr/bin/env bash
# Independent four-GPU capacity/throughput gate for the new global-batch-64
# Math configuration.  It never participates in the main distributed job.
# A failed probe is retained and the next smaller micro-batch is attempted.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20d:4
#SBATCH --nodelist=lj-gpu2
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --output=/historical/shared/qat/r/mopd-logs/mopd-perf-gate-%j.out
#SBATCH --error=/historical/shared/qat/r/mopd-logs/mopd-perf-gate-%j.err
#SBATCH --export=ALL

set -uo pipefail
ROOT=/historical/home/QAT
PROFILE=deepseek_skywork
VLLM_ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export MOPD_VERL_DIR="$ROOT/vendor/verl"
export PATH="$VLLM_ENV/bin:$PATH"
export MOPD_RAY_PATH="$VLLM_ENV/bin:$PATH"
PYTHON_INCLUDE_DIR=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export MOPD_PYTHON_INCLUDE="$PYTHON_INCLUDE_DIR"
export CPATH="$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYTHON_INCLUDE_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p /historical/shared/qat/r/mopd-logs "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

OUT=/historical/shared/qat/runs/mopd-perf/deepseek_skywork/b64-gate-${SLURM_JOB_ID}
mkdir -p "$OUT"
for mb in 16 8 4 2; do
  tag="mb${mb}"
  log="$OUT/${tag}.log"
  echo "[perf-gate] start mb=$mb global_batch=64 ppo_mini=64 $(date -Is)" | tee "$log"
  bash "$ROOT/scripts/run_mopd_verl.sh" \
    --profile "$PROFILE" --rollout-engine vllm \
    --native-turboquant-kv-cache-dtype turboquant_3bit_nc --kv-bits 32 \
    --no-kivi-hf-rollout --sampling-top-k 0 --sampling-top-p 1.0 \
    --stage gate --updates 2 --gpus 4 \
    --train-batch-size 64 --micro-batch-size-per-gpu "$mb" \
    --ppo-mini-batch-size 64 --save-freq 1 --eval-checkpoints 1 \
    --experiment "deepseek_skywork-b64-perf-${tag}-${SLURM_JOB_ID}" \
    --output "$OUT/$tag" --run >>"$log" 2>&1
  rc=$?
  echo "[perf-gate] end mb=$mb rc=$rc $(date -Is)" | tee -a "$log"
done
echo "[perf-gate] complete; logs under $OUT" 
