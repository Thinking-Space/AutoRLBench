#!/usr/bin/env bash
# GRPO | Qwen3-VL-8B-Instruct | geometry3k | FSDP + vLLM
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME
WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Optional shared venv for the selected existing image. In the task Docker
# image, leave this unset and use the image's already-installed environment.
if [[ -n "${QWEN3VL_ENV_DIR:-}" ]]; then
    [[ -x "$QWEN3VL_ENV_DIR/bin/python" ]] || { printf 'Missing environment: %s\n' "$QWEN3VL_ENV_DIR" >&2; exit 2; }
    export VIRTUAL_ENV="$QWEN3VL_ENV_DIR"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    RUNTIME_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    RUNTIME_LIBS="$(python3 - "$RUNTIME_SITE" <<'PY'
from pathlib import Path
import sys
site = Path(sys.argv[1])
print(':'.join([str(site / 'torch/lib')] + [str(p) for p in sorted((site / 'nvidia').rglob('lib')) if p.is_dir()]))
PY
)"
    export LD_LIBRARY_PATH="$RUNTIME_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -x "$RUNTIME_SITE/nvidia/cu13/bin/nvcc" ]]; then
        export CUDA_HOME="$RUNTIME_SITE/nvidia/cu13"
        export PATH="$CUDA_HOME/bin:$PATH"
        export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
    fi
fi
export PYTHONPATH="$WORKSPACE/verl${PYTHONPATH:+:$PYTHONPATH}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-8B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE/data/geo3k/train.parquet}"
TEST_FILE="${TEST_FILE:-$WORKSPACE/data/geo3k/test.parquet}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
mkdir -p "$LOG_DIR"
export WANDB_MODE=offline WANDB_API_KEY=offline
export WANDB_DIR="${WANDB_DIR:-$LOG_DIR}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for asset in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TEST_FILE"; do
    [[ -f "$asset" ]] || { printf 'Missing prepared asset: %s\n' "$asset" >&2; exit 2; }
done
cd "$WORKSPACE/verl"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.image_patch_size=16 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=geometry3k_grpo \
    trainer.experiment_name=qwen3vl_8b_fsdp \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$LOG_DIR/checkpoints" \
    trainer.rollout_data_dir="$LOG_DIR/rollouts" \
    trainer.validation_data_dir="$LOG_DIR/validation" \
    "$@" 2>&1 | tee "$LOG_DIR/train.log"
