#!/bin/bash
#SBATCH --job-name=qat-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=/historical/shared/qat/r/qat-train/train-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-train/train-%j.err
#
# Parameterised TurboQuant-QAT training launcher.
#
# slurm_qat_gate.sbatch is a 2-update smoke with its batch sizes and 4-GPU
# layout baked in.  This script is the same known-good path with the knobs a
# real run needs exposed: batch sizes, GPU count, update count, checkpoint
# frequency, resume, and an FP control mode.
#
# --gres and --time in the directives above are defaults; override them on the
# sbatch command line (sbatch --time=03:00:00 --gres=gpu:nvidia_l20d:8 ...).
#
# Env knobs:
#   GB       data.train_batch_size          (default 64)
#   MINI     actor.ppo_mini_batch_size      (default: = GB, i.e. strictly on-policy)
#   MB       micro batch size per GPU       (default 1)
#   RM_MB    reward-forward micro batch     (default 1)
#   LP_MB    log-prob/distill-reward micro batch (default 1)
#   GPUS     trainer.n_gpus_per_node        (default 4)
#   UPDATES  total optimizer steps          (default 120)
#   SAVE_FREQ  checkpoint every N steps     (default 20)
#   MODE     qat | fp                       (default qat; fp = no quantizer at all)
#   BLOCK_ALIGN_COEF                        (default 0.5; forced to 0 when MODE=fp)
#   RESUME_PATH  absolute path to a global_step_N dir to resume from
#   ENABLE_TORCH_COMPILE=1  opt out of the L20D Triton guard (validated by
#                job 22447: full AIME eval on the compiled path scored
#                0.32188 vs 0.30521 eager, i.e. no regression, at 1.60x)
#   TAG      extra string in the output dir name
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
# The compute nodes have no system ninja; flashinfer JIT-compiles its TRTLLM
# attention module at model-init time and shells out to it.  The venv ships
# one, but only an activated shell puts that bin dir on PATH.
MOPD_VENV_BIN="${MOPD_VENV_BIN:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin}"
export PATH="$MOPD_VENV_BIN:$PATH"
command -v ninja >/dev/null || echo "[qat-train] WARNING: ninja still not on PATH" >&2

PROFILE="${PROFILE:-deepseek_skywork}"
MODEL_PATH="${MODEL_PATH:-}"
TEACHER_PATH="${TEACHER_PATH:-}"
TRAIN_PATH="${TRAIN_PATH:-}"
MODE="${MODE:-qat}"
GB="${GB:-64}"
MINI="${MINI:-$GB}"
MB="${MB:-1}"
# Frozen-teacher reward forward micro-batch.  Inference-only and therefore a
# pure throughput knob (see preflight_mopd.py).  Default 1 keeps older runs
# reproducible bit-for-bit.
RM_MB="${RM_MB:-1}"
# Micro-batch for the two actor-side inferential forwards (compute_log_prob
# and compute_distillation_reward).  See the note on
# --log-prob-micro-batch-size-per-gpu in preflight_mopd.py.
LP_MB="${LP_MB:-1}"
GPUS="${GPUS:-4}"
UPDATES="${UPDATES:-120}"
SAVE_FREQ="${SAVE_FREQ:-20}"
BLOCK_ALIGN_COEF="${BLOCK_ALIGN_COEF:-0.5}"
BLOCK_ALIGN_LAYER_EVERY="${BLOCK_ALIGN_LAYER_EVERY:-4}"
BLOCK_ALIGN_KEEP_FRACTION="${BLOCK_ALIGN_KEEP_FRACTION:-0.125}"
BLOCK_ALIGN_METRIC="${BLOCK_ALIGN_METRIC:-nmse}"
BLOCK_ALIGN_LAST_ONLY="${BLOCK_ALIGN_LAST_ONLY:-0}"
LR="${LR:-1e-6}"
REWARD_TOP_K="${REWARD_TOP_K:-64}"
TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
# Objective split used by the two 1.5B runs.  The actor reads this inside
# Ray workers; keep it explicit in the launcher log and SwanLab run metadata.
QAT_PROBE_OBJECTIVE="${QAT_PROBE_OBJECTIVE:-both}"
export QAT_PROBE_OBJECTIVE
PROJECT_NAME="${PROJECT_NAME:-qat-distill-1p5b}"
# "lo,hi": scale the OPD reward linearly along the response.  Exported so
# dp_actor sees it inside the Ray workers.
export QAT_OPD_POSITION_RAMP="${QAT_OPD_POSITION_RAMP:-}"
export QAT_OPD_POSITION_RAMP_NORMALIZE="${QAT_OPD_POSITION_RAMP_NORMALIZE:-0}"
export QAT_BLOCK_ALIGN_STATIC_REF="${QAT_BLOCK_ALIGN_STATIC_REF:-1}"
# teacher entropy 只用于打日志, 却对每个 response token 做全词表 log_softmax
export QAT_SKIP_TEACHER_ENTROPY="${QAT_SKIP_TEACHER_ENTROPY:-0}"
export QAT_PROBE_OBJECTIVE="${QAT_PROBE_OBJECTIVE:-both}"
export QAT_GRAD_PROBE="${QAT_GRAD_PROBE:-0}"
# Empty = torch FSDP default (FULL_SHARD / ZeRO-3).  SHARD_GRAD_OP is
# ZeRO-2: parameters stay resident between micro-batches.
export QAT_FSDP_SHARDING_STRATEGY="${QAT_FSDP_SHARDING_STRATEGY:-}"
# 数据集 prompt 全量分布 mean=98 p99=318 p999=777 max=1670；
# 512 只过滤 0.31%(49/16000)，而把左填充从 ~1950 降到 ~414。
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-16384}"
TAG="${TAG:-}"

case "$MODE" in
  qat)
    TQ_DTYPE="${TQ_DTYPE:-turboquant_k3v4_nc}"
    KV_SCHEME="${KV_SCHEME:-turboquant_k3v4_nc}"
    SCHEME_ARGS=(--native-turboquant-kv-cache-dtype "$TQ_DTYPE" --kv-scheme "$KV_SCHEME")
    # Distinguish the block-alignment term, the KV-alignment term, or both in
    # the output dir so the two arms never collide.
    TERMS="ba$BLOCK_ALIGN_COEF"
    if [[ -n "${KV_ALIGN_COEF:-}" && "${KV_ALIGN_COEF}" != "0.0" ]]; then
      TERMS="${TERMS}-kva${KV_ALIGN_COEF}"
    fi
    NAME="$KV_SCHEME-$TERMS"
    ;;
  fp)
    # Control arm: no native TurboQuant cache in vLLM and no actor-side
    # fake quantizer.  block-align has nothing to align against, so it is
    # forced off -- leaving it on would abort in dp_actor.
    BLOCK_ALIGN_COEF=0.0
    # preflight requires the frozen-model eval-gate summary for stage=main
    # whenever the native TurboQuant rollout is off (preflight_mopd.py:970).
    # This arm is exactly the diagnostic run that escape hatch is for.
    SCHEME_ARGS=(--native-turboquant-kv-cache-dtype none --allow-missing-eval-gate)
    NAME="fp-control"
    ;;
  wa)
    # NVFP4 weight/activation QAT with an unquantized KV cache (the "w4a4 kv16"
    # tier).  No TurboQuant cache in vLLM and no actor-side KV fake quantizer;
    # the quantization lives in qat_kv.wa_scheme instead.  Unlike MODE=fp,
    # block alignment is *not* forced off: it scores per-block hidden states
    # against a full-precision reference, and NVFP4 W/A perturbs those exactly
    # as KV quantization does, so the NMSE term has a real target.
    : "${QAT_WA_SCHEME:?MODE=wa requires QAT_WA_SCHEME (nvfp4|nvfp4_w4a16)}"
    : "${QAT_WA_STATE:?MODE=wa requires QAT_WA_STATE (modelopt_state.pt from ptq_nvfp4.py)}"
    [[ -f "$QAT_WA_STATE" ]] || { echo "[qat-train] missing QAT_WA_STATE: $QAT_WA_STATE" >&2; exit 2; }
    SCHEME_ARGS=(
      --native-turboquant-kv-cache-dtype none
      --allow-missing-eval-gate
      --wa-scheme "$QAT_WA_SCHEME"
      --wa-state "$QAT_WA_STATE"
    )
    TERMS="ba$BLOCK_ALIGN_COEF"
    NAME="$QAT_WA_SCHEME-kv16-$TERMS"
    ;;
  *)
    echo "[qat-train] unknown MODE=$MODE (expected qat|fp|wa)" >&2
    exit 2
    ;;
esac

MODEL_ARGS=()
[[ -n "$MODEL_PATH" ]] && MODEL_ARGS+=(--model "$MODEL_PATH")
[[ -n "$TEACHER_PATH" ]] && MODEL_ARGS+=(--teacher "$TEACHER_PATH")
[[ -n "$TRAIN_PATH" ]] && MODEL_ARGS+=(--train "$TRAIN_PATH")

mkdir -p /historical/shared/qat/r/qat-train
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
        echo "[qat-train] /tmp has ${_tmp_free_kb:-unknown} KB free, need $TRITON_CACHE_FS_MIN_KB;" >&2
        echo "[qat-train] refusing to put the Triton cache back on /shared (race-prone)." >&2
        echo "[qat-train] Set TRITON_CACHE_DIR explicitly to override." >&2
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
# With torch.compile on, two jobs starting at the same time compile into the
# same inductor cache and clobber each other's half-written modules (job 22685
# died with InductorError: FileNotFoundError on a .py under inductor_cache/).
# Key the cache on the arm so different arms never collide, while a restart of
# the same arm still reuses what it already compiled.
_cache_key="${TAG:-${SLURM_JOB_ID:-shared}}"
if [[ "${ENABLE_TORCH_COMPILE:-0}" == "1" ]]; then
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor-$_cache_key}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm-$_cache_key}"
else
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
fi
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

if [[ "${RESUME_AUTO:-0}" == "1" ]]; then
  # Continue a run that was cut off by its wall clock.  resume_mode=auto reads
  # latest_checkpointed_iteration.txt, so the step number does not have to be
  # known when this job is queued -- which is the point: it can be submitted
  # with --dependency=afterany on a job that is still running.
  : "${RUN_DIR:?RESUME_AUTO=1 requires RUN_DIR}"
  OUT="$RUN_DIR"
  RESUME_ARGS=(--resume)
  echo "[qat-train] AUTO-RESUMING $OUT from its latest checkpoint"
elif [[ -n "${RESUME_PATH:-}" ]]; then
  # Resume writes into the ORIGINAL run directory so the trainer's own
  # bookkeeping (latest_checkpointed_iteration.txt, provenance) stays coherent.
  OUT="$(cd "$RESUME_PATH/../.." && pwd)"
  RESUME_ARGS=(--resume --resume-path "$RESUME_PATH")
  echo "[qat-train] RESUMING from $RESUME_PATH into $OUT"
else
  OUT="/historical/shared/qat/runs/qat-train/${PROFILE}/${NAME}-g${GPUS}-b${GB}-lr${LR}-k${REWARD_TOP_K}-${TOP_K_STRATEGY}${TAG:+-$TAG}-pl${PROMPT_LENGTH}-j${SLURM_JOB_ID}"
  RESUME_ARGS=()
fi

if [[ "${ENABLE_TORCH_COMPILE:-0}" == "1" ]]; then
  COMPILE_ARGS=(--enable-torch-compile)
  echo "[qat-train] torch.compile ENABLED for the vLLM rollout"
else
  COMPILE_ARGS=()
fi

echo "[qat-train] mode=$MODE objective=$QAT_PROBE_OBJECTIVE gpus=$GPUS gb=$GB mini=$MINI mb=$MB updates=$UPDATES save_freq=$SAVE_FREQ"
echo "[qat-train] opd_ramp=${QAT_OPD_POSITION_RAMP:-none} normalize=$QAT_OPD_POSITION_RAMP_NORMALIZE static_ref=$QAT_BLOCK_ALIGN_STATIC_REF"
echo "[qat-train] lr=$LR block_align=$BLOCK_ALIGN_COEF every=$BLOCK_ALIGN_LAYER_EVERY keep=$BLOCK_ALIGN_KEEP_FRACTION metric=$BLOCK_ALIGN_METRIC prompt_len=$PROMPT_LENGTH util=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
echo "[qat-train] output=$OUT"

# store_true flag: it must be passed bare or not at all.  "--flag=false" is a
# parse error, not a disable.
if [[ "$BLOCK_ALIGN_LAST_ONLY" = "1" ]]; then
  BLOCK_ALIGN_LAST_ONLY_ARGS=(--block-align-last-only)
else
  BLOCK_ALIGN_LAST_ONLY_ARGS=()
fi

exec bash "$ROOT/scripts/run_mopd_verl.sh" \
  --profile "$PROFILE" \
  "${MODEL_ARGS[@]}" \
  --rollout-engine vllm \
  "${SCHEME_ARGS[@]}" \
  --no-kivi-hf-rollout \
  --block-align-coef "$BLOCK_ALIGN_COEF" \
  --block-align-layer-every "$BLOCK_ALIGN_LAYER_EVERY" \
  --block-align-keep-fraction "$BLOCK_ALIGN_KEEP_FRACTION" \
  --block-align-metric "$BLOCK_ALIGN_METRIC" \
  "${BLOCK_ALIGN_LAST_ONLY_ARGS[@]}" \
  --kv-align-coef "${KV_ALIGN_COEF:-0.0}" \
  --kv-align-layer-every "${KV_ALIGN_LAYER_EVERY:-1}" \
  --kv-align-keep-fraction "${KV_ALIGN_KEEP_FRACTION:-1.0}" \
  --kv-align-metric "${KV_ALIGN_METRIC:-nmse}" \
  --lr "$LR" \
  --reward-top-k "$REWARD_TOP_K" \
  --top-k-strategy "$TOP_K_STRATEGY" \
  --reward-weight-mode "$REWARD_WEIGHT_MODE" \
  --sampling-top-k 0 \
  --sampling-top-p 1.0 \
  --gpus "$GPUS" \
  --train-batch-size "$GB" \
  --micro-batch-size-per-gpu "$MB" \
  --reward-micro-batch-size-per-gpu "$RM_MB" \
  --log-prob-micro-batch-size-per-gpu "$LP_MB" \
  --ppo-mini-batch-size "$MINI" \
  --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}" \
  --prompt-length "$PROMPT_LENGTH" \
  --response-length "$RESPONSE_LENGTH" \
  --stage main --updates "$UPDATES" \
  --experiment "${PROFILE}-${NAME}${TAG:+-$TAG}-j${SLURM_JOB_ID}" \
  --output "$OUT" \
  --project "$PROJECT_NAME" \
  --save-freq "$SAVE_FREQ" \
  --eval-checkpoints "${EVAL_CHECKPOINTS:-30,60,90,120,150}" \
  "${RESUME_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  --run
