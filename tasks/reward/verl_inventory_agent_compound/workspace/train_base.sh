#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

task_model="${MODEL_PATH:-$PWD/model}"
task_steps="${STEPS:-20}"
task_seed="${SEED:-42}"
task_gpus="${N_GPUS:-1}"
mkdir -p "$PWD/outputs/rollouts" "$PWD/outputs/validation"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export VERL_FILE_LOGGER_PATH="$PWD/outputs/metrics.jsonl"
export VLLM_USE_V1=1

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$PWD/data/train.parquet" \
  data.val_files="$PWD/data/val.parquet" \
  data.train_batch_size=32 \
  data.val_batch_size=128 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.seed="$task_seed" \
  actor_rollout_ref.model.path="$task_model" \
  actor_rollout_ref.model.use_remove_padding=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  reward_model.enable=false \
  reward_model.reward_manager=batch \
  custom_reward_function.path="$PWD/case/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger='[console,file]' \
  trainer.project_name=arithmetic_batch \
  trainer.experiment_name="seed_${task_seed}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$task_gpus" \
  trainer.total_training_steps="$task_steps" \
  trainer.total_epochs=10 \
  trainer.val_before_train=true \
  trainer.test_freq=5 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$PWD/outputs/checkpoints" \
  trainer.rollout_data_dir="$PWD/outputs/rollouts" \
  trainer.validation_data_dir="$PWD/outputs/validation" \
  trainer.log_val_generations=5 \
  "$@" 2>&1 | tee "$PWD/outputs/train.log"
