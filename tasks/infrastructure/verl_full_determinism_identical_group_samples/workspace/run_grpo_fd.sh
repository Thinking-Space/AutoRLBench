#!/bin/bash
# GRPO on GSM8K with Qwen2.5-1.5B-Instruct (verl FSDP + vLLM), sized for one GPU, with verl's full-determinism mode
# enabled as in docs/advance/determinism.md (rollout seed 42; deterministic actor and ref). vLLM 0.17's
# batch-invariant mode, which full_determinism turns on, refuses to start unless the attention backend is set
# explicitly, and torch 2.10's deterministic Inductor rejects vLLM's compile-time autotuning, so the rollout runs
# eager.
# Extra Hydra overrides given on the command line are appended last and take precedence.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"

MODEL=${MODEL:-$WS/models/Qwen2.5-1.5B-Instruct}
DATA=${DATA:-$WS/data}
N_GPUS=${N_GPUS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-30}
TEST_FREQ=${TEST_FREQ:-5}
BATCH=${BATCH:-64}
N_SAMPLES=${N_SAMPLES:-8}
MAX_RESPONSE=${MAX_RESPONSE:-1024}
SEED=${SEED:-1}
EXP=${EXP:-qwen25_1.5b_gsm8k_grpo_deterministic}
LOG_DIR=${LOG_DIR:-$WS/outputs}

mkdir -p "$LOG_DIR"
export PYTHONPATH="$WS/verl${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$LOG_DIR"
export RAY_DEDUP_LOGS=0

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/gsm8k/train.parquet" \
  data.val_files="$DATA/gsm8k/test.parquet" \
  data.train_batch_size=$BATCH \
  data.max_prompt_length=512 \
  data.max_response_length=$MAX_RESPONSE \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed=$SEED \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$((BATCH / 2)) \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.full_determinism=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$N_SAMPLES \
  actor_rollout_ref.rollout.full_determinism=True \
  actor_rollout_ref.rollout.seed=42 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.full_determinism=True \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=gsm8k_grpo_deterministic \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=-1 \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.default_local_dir="$LOG_DIR/checkpoints" \
  trainer.rollout_data_dir="$LOG_DIR/rollouts" \
  trainer.validation_data_dir="$LOG_DIR/val_generations" \
  ray_kwargs.ray_init.num_cpus=32 \
  +ray_kwargs.ray_init.object_store_memory=21474836480 \
  "$@" 2>&1 | tee "$LOG_DIR/train.log"
