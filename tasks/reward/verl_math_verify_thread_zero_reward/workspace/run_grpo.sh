#!/bin/bash
# GRPO on a GSM8K + MATH mixture with Qwen2.5-1.5B-Instruct (verl FSDP + vLLM), validated on GSM8K test + MATH-500.
# Extra Hydra overrides given on the command line are appended last and take precedence.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"

MODEL=${MODEL:-$WS/models/Qwen2.5-1.5B-Instruct}
DATA=${DATA:-$WS/data}
TRAIN_FILES=${TRAIN_FILES:-"['$DATA/gsm8k/train.parquet','$DATA/math/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['$DATA/gsm8k/test.parquet','$DATA/math500/test.parquet']"}
N_GPUS=${N_GPUS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-5}
BATCH=${BATCH:-64}
N_SAMPLES=${N_SAMPLES:-8}
MAX_RESPONSE=${MAX_RESPONSE:-3072}
SEED=${SEED:-1}
EXP=${EXP:-qwen25_1.5b_gsm8k_math_grpo}
LOG_DIR=${LOG_DIR:-$WS/outputs}

mkdir -p "$LOG_DIR"
# third_party/python holds math-verify (used by reward/mixed_math_reward.py), installed by the prepare step.
export PYTHONPATH="$WS/verl:$WS/third_party/python${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$LOG_DIR"
export RAY_DEDUP_LOGS=0

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
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
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$N_SAMPLES \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$WS/reward/mixed_math_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=gsm8k_math_grpo \
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
  ray_kwargs.ray_init.num_cpus=48 \
  +ray_kwargs.ray_init.object_store_memory=32212254720 \
  "$@" 2>&1 | tee "$LOG_DIR/train.log"
