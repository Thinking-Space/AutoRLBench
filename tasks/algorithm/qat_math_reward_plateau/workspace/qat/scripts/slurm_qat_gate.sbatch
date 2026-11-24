#!/bin/bash
#SBATCH --job-name=qat-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --output=/historical/shared/qat/r/qat-gate/gate-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-gate/gate-%j.err
#
# Minimal launcher for the TurboQuant-QAT gate.
#
# Deliberately NOT slurm_mopd_native_tq.sbatch: that script's MODE=smoke branch
# has never been observed to reach a training step (7161/7187/7188 all stalled
# before TaskRunner, and a no-QAT control 7189 stalled identically, so the
# stall is not QAT-related), and its main branch pins a fixed output directory
# that preflight rightly refuses to overwrite.  It also auto-submits an eval
# watcher, which keeps resurrecting evaluation jobs.
#
# This script does one thing: a few real optimizer updates with the actor
# fake-quantizer installed, into a fresh per-job output directory.
set -euo pipefail

# The ccremote agent shell exports LD_PRELOAD=libproxychains4.so; with
# --export=ALL that shim is injected into every job process and hangs Ray
# actor startup (TaskRunner never connects).  Strip it unconditionally.
unset LD_PRELOAD

ROOT=/historical/home/QAT
# Without this, run_mopd_verl.sh falls back to the reference copy under
# /shared/.../Open-MOPD/training/verl and none of the vendored verl edits
# (block-align term, ActorConfig fields, actor.yaml keys) are loaded.
export MOPD_VERL_DIR="$ROOT/vendor/verl"
# The compute nodes have no system ninja; flashinfer JIT-compiles its
# TRTLLM attention module at model-init time and shells out to it.  The
# venv ships one, but only an activated shell puts that bin dir on PATH --
# runs submitted from a non-activated shell fail with
# FileNotFoundError: 'ninja'.  Prepend it unconditionally.
MOPD_VENV_BIN="${MOPD_VENV_BIN:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin}"
export PATH="$MOPD_VENV_BIN:$PATH"
command -v ninja >/dev/null || echo "[qat-gate] WARNING: ninja still not on PATH" >&2
PROFILE="${PROFILE:-deepseek_skywork}"
TQ_DTYPE="${TQ_DTYPE:-turboquant_k3v4_nc}"
KV_SCHEME="${KV_SCHEME:-turboquant_k3v4_nc}"
BLOCK_ALIGN_COEF="${BLOCK_ALIGN_COEF:-0.5}"
BLOCK_ALIGN_LAYER_EVERY="${BLOCK_ALIGN_LAYER_EVERY:-1}"
BLOCK_ALIGN_KEEP_FRACTION="${BLOCK_ALIGN_KEEP_FRACTION:-1.0}"
BLOCK_ALIGN_METRIC="${BLOCK_ALIGN_METRIC:-nmse}"
KV_ALIGN_COEF="${KV_ALIGN_COEF:-0.0}"
UPDATES="${UPDATES:-2}"
MB="${MB:-2}"
# 数据集 prompt 全量分布 mean=98 p99=318 p999=777 max=1670；
# 512 只过滤 0.31%(49/16000)，而把左填充从 ~1950 降到 ~414。
# 不要用 256：那会静默丢掉 2.03% 的样本。
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
GB="${GB:-8}"

mkdir -p /historical/shared/qat/r/qat-gate
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
# Compute nodes have no python3.12-devel, so Triton cannot build its launcher
# stub and every JIT miss dies with "Python.h: No such file or directory".
# These runs only survived on a warm shared Triton cache; a new kernel (int4)
# is a guaranteed miss.  Same vendored tree the other launchers use.
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PYINC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
# Node-local, per-job Triton cache.
#
# Triton publishes a compiled kernel as several sibling files and relies on
# file locking to keep concurrent compilers from reading a half-published
# entry.  That locking does not hold on /shared, so the workers race:
# jobs 12089/12090 died ~4 min in on a missing .source and a missing
# __triton_launcher .so.  Moving to a per-job directory on /shared was not
# enough -- job 12192 hit the same missing .so inside its own private
# job-12192/ dir, because the four rollout workers of a single job compile the
# same kernels at the same time.  The race is between processes, not jobs, so
# the fix has to be a filesystem where flock actually works: node-local /tmp.
#
# The cost is one recompile per job (~1-2 min) and a cache that dies with the
# node allocation.  A single job's cache is well under 100 MB.
TRITON_CACHE_FS_MIN_KB=1048576  # 1 GiB
if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
    # /tmp on these nodes is a shared local NVMe that has been seen at 98%
    # full.  Silently spilling back to /shared would reintroduce exactly the
    # race this block exists to remove, so run out of space loudly instead.
    _tmp_free_kb="$(df -Pk /tmp 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -z "$_tmp_free_kb" || "$_tmp_free_kb" -lt "$TRITON_CACHE_FS_MIN_KB" ]]; then
        echo "[qat-gate] /tmp has ${_tmp_free_kb:-unknown} KB free, need $TRITON_CACHE_FS_MIN_KB;" >&2
        echo "[qat-gate] refusing to put the Triton cache back on /shared (race-prone)." >&2
        echo "[qat-gate] Set TRITON_CACHE_DIR explicitly to override." >&2
        exit 2
    fi
    TRITON_CACHE_DIR="/tmp/qat-triton-${SLURM_JOB_ID:-$$}"
fi
export TRITON_CACHE_DIR
# This script ends in `exec`, so an EXIT trap would never fire -- the cache has
# to be reaped on the way in instead.  Only sweep dirs older than a day: a
# younger one may belong to a job still running on this node.
find /tmp -maxdepth 1 -name 'qat-triton-*' -type d -mtime +1 \
    -exec rm -rf {} + 2>/dev/null || true
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

OUT="/historical/shared/qat/runs/qat-gate/${PROFILE}/${KV_SCHEME}-ba${BLOCK_ALIGN_COEF}${KV_ALIGN_COEF:+-kva$KV_ALIGN_COEF}${PROMPT_LENGTH:+-pl$PROMPT_LENGTH}-j${SLURM_JOB_ID}"

echo "[qat-gate] scheme=$KV_SCHEME dtype=$TQ_DTYPE block_align=$BLOCK_ALIGN_COEF kv_align=$KV_ALIGN_COEF updates=$UPDATES"
echo "[qat-gate] output=$OUT"

exec bash "$ROOT/scripts/run_mopd_verl.sh" \
  --profile "$PROFILE" \
  --rollout-engine vllm \
  --native-turboquant-kv-cache-dtype "$TQ_DTYPE" \
  --no-kivi-hf-rollout \
  --kv-scheme "$KV_SCHEME" \
  --block-align-coef "$BLOCK_ALIGN_COEF" \
  --block-align-layer-every "$BLOCK_ALIGN_LAYER_EVERY" \
  --block-align-keep-fraction "$BLOCK_ALIGN_KEEP_FRACTION" \
  --block-align-metric "$BLOCK_ALIGN_METRIC" \
  --kv-align-coef "$KV_ALIGN_COEF" \
  --sampling-top-k 0 \
  --sampling-top-p 1.0 \
  --gpus 4 \
  --train-batch-size "$GB" \
  --micro-batch-size-per-gpu "$MB" \
  --ppo-mini-batch-size "$GB" \
  --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}" \
  ${PROMPT_LENGTH:+--prompt-length "$PROMPT_LENGTH"} \
  --stage gate --updates "$UPDATES" \
  --experiment "${PROFILE}-qatgate-j${SLURM_JOB_ID}" \
  --output "$OUT" \
  --save-freq "$UPDATES" \
  --eval-checkpoints "$UPDATES" \
  --run
