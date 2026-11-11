#!/usr/bin/env bash
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$W/scripts:$W/verl${PYTHONPATH:+:$PYTHONPATH}"
export RUN_DIR="$W/runs/run_001"
export ATTEMPT="$(date -u +%Y%m%dT%H%M%S)-$$"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export TMPDIR="${TMPDIR:-$W/runtime/tmp}"
export XDG_CACHE_HOME="$W/runtime/cache"
export HF_HOME="$W/runtime/hf"
export TRITON_CACHE_DIR="$W/runtime/triton"
export TORCHINDUCTOR_CACHE_DIR="$W/runtime/inductor"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
export RAY_TMPDIR="${RAY_TMPDIR:-$(mktemp -d "$TMPDIR/ray.XXXXXX")}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export RAY_DEDUP_LOGS=0
export VLLM_USE_V1=0
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/validation"
cd "$W/verl"
python3 -u "$W/scripts/train_entry.py" \
  "data.train_files=$W/data/train.parquet" \
  "data.val_files=[$W/data/aime2024.parquet,$W/data/aime2025.parquet]" \
  "actor_rollout_ref.model.path=$W/model" \
  "trainer.default_local_dir=$RUN_DIR/checkpoints" \
  "trainer.validation_data_dir=$RUN_DIR/validation" \
  trainer.resume_mode=auto trainer.del_local_ckpt_after_load=False \
  "hydra.run.dir=$RUN_DIR/hydra/$ATTEMPT" hydra.output_subdir=.hydra \
  "$@" 2>&1 | tee "$RUN_DIR/logs/$ATTEMPT.log"
