#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_FILE_LOGGER_PATH="$PWD/outputs/metrics.jsonl"
export CASE_SYNC_TRACE="$PWD/outputs/sync_trace.jsonl"
mkdir -p outputs
exec python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files="$PWD/data/train.parquet" data.val_files="$PWD/data/val.parquet" \
 data.train_batch_size=32 data.max_prompt_length=256 data.max_response_length=192 \
 data.filter_overlong_prompts=false data.truncation=error data.seed="${SEED:-42}" \
 +data.apply_chat_template_kwargs.enable_thinking=false \
 actor_rollout_ref.model.path="${MODEL_PATH:-$PWD/model}" \
 actor_rollout_ref.model.use_remove_padding=false \
 actor_rollout_ref.model.enable_gradient_checkpointing=true \
 +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
 actor_rollout_ref.actor.optim.lr=0.000003 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.8 \
 actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.free_cache_engine=true \
 actor_rollout_ref.rollout.max_model_len=448 actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=false actor_rollout_ref.rollout.val_kwargs.n=1 \
 reward_model.reward_manager=naive custom_reward_function.path="$PWD/case/reward.py" custom_reward_function.name=compute_score \
 algorithm.use_kl_in_reward=false \
 trainer.logger='[console,file]' trainer.project_name=autorlbench_candidates trainer.experiment_name=rollout_sync \
 trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=100 trainer.total_training_steps="${STEPS:-20}" \
 trainer.val_before_train=true trainer.test_freq=5 trainer.save_freq=-1 \
 trainer.default_local_dir="$PWD/outputs/checkpoints" \
 trainer.validation_data_dir="$PWD/outputs/validation" trainer.rollout_data_dir="$PWD/outputs/rollouts" \
 "$@"
