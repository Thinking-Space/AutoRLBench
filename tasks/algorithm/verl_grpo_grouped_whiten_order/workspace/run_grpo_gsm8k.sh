#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN_NAME=${RUN_NAME:-grpo_gsm8k}
MODEL_PATH=${MODEL_PATH:-$PWD/models/Qwen2.5-0.5B-Instruct}
RAW_DATA=${RAW_DATA:-$PWD/raw/gsm8k}
DATA_DIR=${DATA_DIR:-$PWD/data/gsm8k}
OUT=${OUT:-$PWD/results/$RUN_NAME}
STEPS=${STEPS:-50}
SEED=${SEED:-42}
mkdir -p "$OUT"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export WANDB_MODE=offline WANDB_DIR=$OUT WANDB_DISABLE_CODE=true
export VERL_FILE_LOGGER_PATH=$OUT/metrics.jsonl
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
python3 data_prep/gsm8k.py --src "$RAW_DATA" --out "$DATA_DIR" 2>&1 | tee "$OUT/data_prep.log"
python3 -m verl.trainer.main_ppo \
  ray_kwargs.ray_init.num_cpus=12 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=64 \
  data.max_prompt_length=384 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed="$SEED" \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=2.5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.data_loader_seed="$SEED" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
  reward.custom_reward_function.path="$PWD/reward/gsm8k_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  trainer.critic_warmup=0 \
  trainer.logger='["console","file","wandb"]' \
  trainer.project_name=math_grpo \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=25 \
  trainer.val_before_train=True \
  trainer.total_training_steps="$STEPS" \
  trainer.rollout_data_dir="$OUT/rollouts" \
  trainer.validation_data_dir="$OUT/val_generations" \
  trainer.default_local_dir="$OUT/checkpoints" \
  "$@" 2>&1 | tee "$OUT/train.log"
