#!/usr/bin/env bash
# One full-node launcher for the native-vLLM-TurboQuant VeRL OPD profiles.
# The wrapper remains the source of truth for the VeRL/Hydra command; this
# file only supplies a reproducible Slurm resource envelope.
#
# Examples (submit only after the two base evaluations have finished):
#   sbatch --export=ALL,PROFILE=deepseek_skywork scripts/slurm_mopd_native_tq.sbatch
#   sbatch --export=ALL,PROFILE=qwen25_longrlvr scripts/slurm_mopd_native_tq.sbatch
#   sbatch --export=ALL,PROFILE=deepseek_skywork,MODE=smoke scripts/slurm_mopd_native_tq.sbatch
#   sbatch --export=ALL,PROFILE=qwen25_longrlvr,MODE=load,LOAD_MICRO_BATCH_SIZE_PER_GPU=4 scripts/slurm_mopd_native_tq.sbatch
#
# MODE=capacity probes descending per-GPU micro-batches and, after the first
# successful real update, continues in the formal 64/200-step run using the
# largest stable value. MODE=smoke/load remain one-update diagnostics.
# MODE=resume continues an interrupted formal run from RESUME_OUTPUT.
# No custom RL loop is implemented here.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 使用任意单节点的 4 张 L20D GPU；Slurm 选择最早可用的节点。
#SBATCH --gres=gpu:nvidia_l20d:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=7-00:00:00
#SBATCH --output=/historical/shared/qat/r/mopd-logs/%x-%j.out
#SBATCH --error=/historical/shared/qat/r/mopd-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

# The ccremote agent shell exports LD_PRELOAD=libproxychains4.so; with
# --export=ALL that shim is injected into every job process and hangs Ray
# actor startup (TaskRunner never connects).  Strip it unconditionally.
unset LD_PRELOAD

: "${PROFILE:?set PROFILE=deepseek_skywork or qwen25_longrlvr}"
MODE="${MODE:-main}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
# Production defaults stay at the conservative value until the profile's
# real-context capacity probe passes.  The probe or a resumed run can override
# this without editing the launcher, e.g. FORMAL_MICRO_BATCH_SIZE_PER_GPU=2.
FORMAL_MICRO_BATCH_SIZE_PER_GPU="${FORMAL_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Formal-run length.  Kept as a variable so a shorter probe run (e.g. a
# 150-step scheme comparison) does not need three edits that could drift
# apart; the default reproduces the previous hard-coded 200.
UPDATES="${UPDATES:-200}"
# Validate against the profile table rather than a private allow-list; the two
# drifted apart and a profile could exist while every launcher rejected it.
if ! "$VLLM_ENV/bin/python" "$ROOT/scripts/profile_env.py" "$PROFILE" >/dev/null; then
  exit 2
fi
case "$MODE" in
  capacity|smoke|load|main|resume) ;;
  *) echo "unsupported MODE=$MODE" >&2; exit 2 ;;
esac

ROOT=/historical/home/QAT
# Use the workspace-pinned VeRL overlay so fixes are versioned and testable;
# the archived shared checkout remains untouched as a reference copy.
export MOPD_VERL_DIR="${MOPD_VERL_DIR:-$ROOT/vendor/verl}"
VLLM_ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
# FlashInfer's sampler JIT invokes the external ``ninja`` binary.  The binary
# is installed in the same environment as vLLM; make it visible to subprocess
# workers as well as to the launcher itself.
export PATH="$VLLM_ENV/bin:$PATH"
export MOPD_RAY_PATH="$VLLM_ENV/bin:$PATH"
# Triton builds a tiny CUDA helper at vLLM startup.  The cluster Python
# runtime has no system Python.h, but the matching 3.12 development headers
# are available in the shared QAT toolchain; expose them to Ray workers too.
PYTHON_INCLUDE_DIR="/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12"
export MOPD_PYTHON_INCLUDE="$PYTHON_INCLUDE_DIR"
export CPATH="$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYTHON_INCLUDE_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
# The installed vLLM/FlashInfer wheels cannot JIT-compile the sampler in this
# Python environment (Python.h is unavailable). This only selects the
# PyTorch sampler; vLLM still owns the TurboQuant KV cache and rollout.
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /historical/shared/qat/r/mopd-logs
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/historical/shared/qat/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# TQ_DTYPE selects the rollout cache dtype; KV_SCHEME turns on the matching
# training-side fake quantizer (QAT).  Preflight refuses --run if the two name
# different presets, so they are kept together here.
TQ_DTYPE="${TQ_DTYPE:-turboquant_3bit_nc}"

COMMON=(
  --profile "$PROFILE"
  --rollout-engine vllm
  --native-turboquant-kv-cache-dtype "$TQ_DTYPE"
  --kv-bits 32
  --no-kivi-hf-rollout
  --sampling-top-k 0
  --sampling-top-p 1.0
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --micro-batch-size-per-gpu "$FORMAL_MICRO_BATCH_SIZE_PER_GPU"
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE"
  --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
  --save-freq 20
  --eval-checkpoints "${EVAL_CHECKPOINTS:-20,40,60,80,100,120,140,160,180,200}"
)

# QAT is opt-in: with KV_SCHEME unset the actor forward stays full precision
# and the run is byte-for-byte the previous configuration.
if [[ -n "${KV_SCHEME:-}" ]]; then
  COMMON+=(--kv-scheme "$KV_SCHEME")
fi
if [[ -n "${BLOCK_ALIGN_COEF:-}" ]]; then
  COMMON+=(--block-align-coef "$BLOCK_ALIGN_COEF")
  [[ -n "${BLOCK_ALIGN_LAYER_EVERY:-}" ]] && COMMON+=(--block-align-layer-every "$BLOCK_ALIGN_LAYER_EVERY")
  [[ -n "${BLOCK_ALIGN_KEEP_FRACTION:-}" ]] && COMMON+=(--block-align-keep-fraction "$BLOCK_ALIGN_KEEP_FRACTION")
  [[ -n "${BLOCK_ALIGN_METRIC:-}" ]] && COMMON+=(--block-align-metric "$BLOCK_ALIGN_METRIC")
fi
if [[ -n "${KV_ALIGN_COEF:-}" ]]; then
  COMMON+=(--kv-align-coef "$KV_ALIGN_COEF")
  [[ -n "${KV_ALIGN_LAYER_EVERY:-}" ]] && COMMON+=(--kv-align-layer-every "$KV_ALIGN_LAYER_EVERY")
  [[ -n "${KV_ALIGN_KEEP_FRACTION:-}" ]] && COMMON+=(--kv-align-keep-fraction "$KV_ALIGN_KEEP_FRACTION")
  [[ -n "${KV_ALIGN_METRIC:-}" ]] && COMMON+=(--kv-align-metric "$KV_ALIGN_METRIC")
  [[ -n "${KV_ALIGN_ALIGN_KEYS:-}" ]] && COMMON+=(--kv-align-align-keys)
  [[ -n "${KV_ALIGN_ALIGN_VALUES:-}" ]] && COMMON+=(--kv-align-align-values)
fi

if [[ "$MODE" == smoke || "$MODE" == load ]]; then
  # One profile-real-context update with a small global batch validates Ray,
  # native cache dtype, rollout log-probs/top-64 transport, teacher scoring,
  # and a finite VeRL optimizer step before the expensive main run.  The
  # default probe uses micro-batch 2 (8 global samples on this 4-GPU run) to
  # measure whether the node can safely carry more than the conservative
  # production setting; set SMOKE_MICRO_BATCH_SIZE_PER_GPU=1 for a fallback.
  if [[ "$MODE" == load ]]; then
    SMOKE_MICRO_BATCH_SIZE_PER_GPU="${LOAD_MICRO_BATCH_SIZE_PER_GPU:?set LOAD_MICRO_BATCH_SIZE_PER_GPU, e.g. 2 or 4}"
  else
    SMOKE_MICRO_BATCH_SIZE_PER_GPU="${SMOKE_MICRO_BATCH_SIZE_PER_GPU:-2}"
  fi
  SMOKE_TRAIN_BATCH_SIZE=$((TRAIN_GPUS * SMOKE_MICRO_BATCH_SIZE_PER_GPU))
  UTIL_TAG="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
  UTIL_TAG="${UTIL_TAG//./p}"
  # Include the Slurm id so a failed probe never makes a retry fail merely
  # because its output directory is non-empty.  Every smoke remains auditable
  # and no previous diagnostics are overwritten.
  PROBE_TAG="${MODE}-mb${SMOKE_MICRO_BATCH_SIZE_PER_GPU}-u${UTIL_TAG}-j${SLURM_JOB_ID}"
  PROBE_OUTPUT="/historical/shared/qat/runs/mopd-capacity/${PROFILE}/${PROBE_TAG}"
  exec bash "$ROOT/scripts/run_mopd_verl.sh" \
    "${COMMON[@]}" --stage gate --updates 1 \
    --gpus "$TRAIN_GPUS" \
    --experiment "${PROFILE}-${PROBE_TAG}" --output "$PROBE_OUTPUT" \
    --train-batch-size "$SMOKE_TRAIN_BATCH_SIZE" \
    --micro-batch-size-per-gpu "$SMOKE_MICRO_BATCH_SIZE_PER_GPU" --ppo-mini-batch-size 8 \
    --eval-checkpoints 1 --save-freq 1 --run
fi

if [[ "$MODE" == capacity ]]; then
  if [[ "$PROFILE" != deepseek_skywork ]]; then
    echo "MODE=capacity is reserved for the Math profile" >&2
    exit 2
  fi
  # With global batch and PPO mini-batch both 64, candidates up to 16 are
  # compatible with the four-GPU world. Probe largest first and let the
  # actual 16K context run determine the stable value.
  CAPACITY_MICRO_BATCHES_PER_GPU="${CAPACITY_MICRO_BATCHES_PER_GPU:-16,8,4,2}"
  CAPACITY_GATE_UPDATES="${CAPACITY_GATE_UPDATES:-2}"
  IFS=',' read -r -a CAPACITY_VALUES <<< "$CAPACITY_MICRO_BATCHES_PER_GPU"
  CAPACITY_ROOT="/historical/shared/qat/runs/mopd-capacity/${PROFILE}/capacity-${SLURM_JOB_ID}"
  mkdir -p "$CAPACITY_ROOT"
  selected_mb=""
  for mb in "${CAPACITY_VALUES[@]}"; do
    [[ "$mb" =~ ^[1-9][0-9]*$ ]] || { echo "invalid micro-batch: $mb" >&2; exit 2; }
    probe_tag="capacity-mb${mb}-j${SLURM_JOB_ID}"
    probe_output="$CAPACITY_ROOT/$probe_tag"
    echo "[capacity] testing micro_batch_size_per_gpu=$mb (global batch=$TRAIN_BATCH_SIZE, PPO mini-batch=$PPO_MINI_BATCH_SIZE)"
    set +e
    bash "$ROOT/scripts/run_mopd_verl.sh" \
      "${COMMON[@]}" --stage gate --updates "$CAPACITY_GATE_UPDATES" \
      --gpus "$TRAIN_GPUS" \
      --experiment "${PROFILE}-${probe_tag}" --output "$probe_output" \
      --train-batch-size "$TRAIN_BATCH_SIZE" \
      --micro-batch-size-per-gpu "$mb" --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
      --eval-checkpoints 1 --save-freq 1 --run \
      >"$CAPACITY_ROOT/$probe_tag.log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      probe_log="$CAPACITY_ROOT/$probe_tag.log"
      # A zero process exit code is not sufficient: the probe must complete
      # multiple consecutive updates to catch memory retained between steps.
      # still report a non-finite actor gradient. Only a real, finite actor
      # update may unlock the formal run; disabled critic metrics are ignored.
      if ! rg -qi "training/global_step:$CAPACITY_GATE_UPDATES" "$probe_log"; then
        echo "[capacity] micro-batch $mb did not record global_step=$CAPACITY_GATE_UPDATES; trying the next lower value"
        continue
      fi
      if ! rg -q 'Open-MOPD veRL source: .*/QAT/vendor/verl/' "$probe_log"; then
        echo "[capacity] micro-batch $mb used an unpinned VeRL source; rejecting stale run"
        continue
      fi
      if rg -qi 'actor/(grad_norm|pg_loss|loss).*nan|actor/(grad_norm|pg_loss|loss).*inf|grad_norm is not finite|Traceback|AssertionError|OutOfMemory|CUDA out of memory' "$probe_log"; then
        echo "[capacity] micro-batch $mb has non-finite actor metrics or an error; trying the next lower value"
        continue
      fi
      selected_mb="$mb"
      echo "$selected_mb" > "$CAPACITY_ROOT/selected_micro_batch_per_gpu"
      echo "[capacity] selected micro_batch_size_per_gpu=$selected_mb"
      break
    fi
    echo "[capacity] micro-batch $mb failed (rc=$rc); trying the next lower value"
  done
  if [[ -z "$selected_mb" ]]; then
    echo "[capacity] no tested micro-batch completed a real VeRL update" >&2
    exit 1
  fi

  # Keep the same allocation and launch formal training immediately after the
  # successful gate. Global batch remains 64; preflight derives accumulation
  # from the selected micro-batch and four-GPU world size.
  FORMAL_OUTPUT="/historical/shared/qat/runs/mopd/${PROFILE}/main-capacity-${SLURM_JOB_ID}"
  export RUN_ROOT="$FORMAL_OUTPUT"
  # Run the watcher as a CPU-only Slurm job.  A child of this batch shell is
  # reaped when the trainer below is launched with ``exec``.
  sbatch --parsable \
    --export=ALL,PROFILE="$PROFILE",RUN_ROOT="$FORMAL_OUTPUT" \
    "$ROOT/scripts/slurm_mopd_eval_watch.sbatch" \
    >"/historical/shared/qat/r/mopd-logs/${SLURM_JOB_ID}-${PROFILE}-eval-watch-submit.log"
  exec bash "$ROOT/scripts/run_mopd_verl.sh" \
    "${COMMON[@]}" --stage main --updates "$UPDATES" --run \
    --gpus "$TRAIN_GPUS" \
    --micro-batch-size-per-gpu "$selected_mb" --train-batch-size "$TRAIN_BATCH_SIZE" \
    --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
    --experiment "${PROFILE}-main-capacity-${SLURM_JOB_ID}" \
    --output "$FORMAL_OUTPUT"
fi

if [[ "$MODE" == resume ]]; then
  if [[ "$PROFILE" != deepseek_skywork ]]; then
    echo "MODE=resume is reserved for the Math profile" >&2
    exit 2
  fi
  RESUME_OUTPUT="${RESUME_OUTPUT:?set RESUME_OUTPUT to the interrupted formal run directory}"
  if [[ ! -d "$RESUME_OUTPUT/checkpoints" ]]; then
    echo "resume output has no checkpoints directory: $RESUME_OUTPUT" >&2
    exit 1
  fi
  if [[ ! -s "$RESUME_OUTPUT/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    echo "resume output has no latest checkpoint marker: $RESUME_OUTPUT" >&2
    exit 1
  fi
  # The existing watcher already monitors this output directory.  Reuse the
  # same VeRL output and let trainer.resume_mode=auto pick its latest complete
  # checkpoint; --resume keeps all prior checkpoints and provenance intact.
  exec bash "$ROOT/scripts/run_mopd_verl.sh" \
    "${COMMON[@]}" --stage main --updates "$UPDATES" --run \
    --gpus "$TRAIN_GPUS" \
    --micro-batch-size-per-gpu "$FORMAL_MICRO_BATCH_SIZE_PER_GPU" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
    --experiment "${PROFILE}-main-resume-${SLURM_JOB_ID}" \
    --output "$RESUME_OUTPUT" --resume
fi

# Keep external evaluation on separate Slurm jobs so the four training GPUs
# remain dedicated to VeRL.  The watcher only wakes every five minutes and
# submits the 3bit TurboQuant evaluation after each complete checkpoint.
# Keep the evaluator outside the training shell in a CPU-only Slurm job.
sbatch --parsable \
  --export=ALL,PROFILE="$PROFILE",RUN_ROOT="/historical/shared/qat/runs/mopd/${PROFILE}/kv32/main200" \
  "$ROOT/scripts/slurm_mopd_eval_watch.sbatch" \
  >"/historical/shared/qat/r/mopd-logs/${SLURM_JOB_ID}-${PROFILE}-eval-watch-submit.log"

exec bash "$ROOT/scripts/run_mopd_verl.sh" \
  "${COMMON[@]}" --stage main --updates "$UPDATES" --run \
  --gpus "$TRAIN_GPUS"
