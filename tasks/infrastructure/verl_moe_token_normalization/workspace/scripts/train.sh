#!/usr/bin/env bash
# Run INSIDE the declared derived image, with workspace mounted writable.
set -euo pipefail
export W=${AUTORLBENCH_WORKSPACE:-$(cd -- "$(dirname -- "$0")/.." && pwd -P)}
export VARIANT=$(cat "$W/variant.txt")
if [[ -n "${AUTORLBENCH_RESULTS:-}" ]]; then
  default_run_dir="$AUTORLBENCH_RESULTS/experiment_logs"
else
  default_run_dir="$W/runs/$VARIANT-$(date +%Y%m%d-%H%M%S)-$$"
fi
export RUN_DIR=${TRAINING_RUN_DIR:-$default_run_dir}
mkdir -p "$RUN_DIR"/{logs,home,tmp,wandb/config,wandb/cache,wandb/data}
mkdir -p "$W/.cache"/{huggingface,torch,torch_extensions,triton,cuda,vllm}
export HOME="$RUN_DIR/home" TMPDIR="$RUN_DIR/tmp" XDG_CACHE_HOME="$W/.cache"
export PYTHONPATH="$W/src/$VARIANT:$W/scripts" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TRAINING_RUNTIME_HOOK=1
export HF_HOME="$W/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$W/.cache/torch" TORCH_EXTENSIONS_DIR="$W/.cache/torch_extensions"
export TRITON_CACHE_DIR="$W/.cache/triton" CUDA_CACHE_PATH="$W/.cache/cuda" VLLM_CACHE_ROOT="$W/.cache/vllm"
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR/wandb" WANDB_CONFIG_DIR="$RUN_DIR/wandb/config"
export WANDB_CACHE_DIR="$RUN_DIR/wandb/cache" WANDB_DATA_DIR="$RUN_DIR/wandb/data"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
PYTHON=${TRAINING_PYTHON:-/usr/bin/python3.12}
cd "$W"
cp "$W/configs/real_gsm8k.yaml" "$RUN_DIR/config-input.yaml"
printf '%q ' "$0" "$@" > "$RUN_DIR/invocation.txt"
printf '\n' >> "$RUN_DIR/invocation.txt"
echo "Run directory: $RUN_DIR"
set +e
"$PYTHON" -m verl.trainer.main_ppo --config-path "$W/configs" --config-name real_gsm8k \
  trainer.total_training_steps=150 trainer.total_epochs=1 trainer.val_before_train=false \
  trainer.test_freq=100 trainer.save_freq=100 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
  "$@" 2>&1 | tee "$RUN_DIR/logs/train.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$RUN_DIR/exit-code.txt"
"$PYTHON" "$W/scripts/export_wandb.py" "$RUN_DIR" > "$RUN_DIR/logs/wandb-export.log" 2>&1
exit "$rc"
