#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
MODEL_PATH="${MODEL_PATH:-$PWD/model}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VERL_FILE_LOGGER_PATH="$PWD/outputs/metrics.jsonl"
mkdir -p outputs

"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$PWD/data/train.parquet" \
  data.val_files="$PWD/data/val.parquet" \
  data.train_batch_size=16 \
  data.val_batch_size=32 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.dataloader_num_workers=0 \
  data.seed="$SEED" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.external_lib=case.loss \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=luspo_token \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len=384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  custom_reward_function.path="$PWD/case/reward.py" \
  custom_reward_function.name=compute_score \
  reward_model.enable=False \
  trainer.logger='[console,file]' \
  trainer.project_name=autorlbench_candidates \
  trainer.experiment_name=arithmetic_length_weighted \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs="$STEPS" \
  trainer.total_training_steps="$STEPS" \
  trainer.test_freq=5 \
  trainer.save_freq=-1 \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$PWD/outputs/checkpoints" \
  trainer.rollout_data_dir="$PWD/outputs/rollouts" \
  trainer.validation_data_dir="$PWD/outputs/validation" \
  trainer.log_val_generations=4 \
  "$@" 2>&1 | tee outputs/train.log
