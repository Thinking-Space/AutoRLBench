#!/usr/bin/env bash
# GRPO on GSM8K with Qwen2.5-0.5B-Instruct, veRL 0.8.0, one GPU.
# Extra Hydra overrides can be appended as arguments.
set -euo pipefail
cd "$(dirname "$0")"
RUN_NAME=${RUN_NAME:-grpo_gsm8k}
MODEL_PATH=${MODEL_PATH:-$PWD/models/Qwen2.5-0.5B-Instruct}
RAW_DATA=${RAW_DATA:-$PWD/raw/gsm8k}
DATA_DIR=${DATA_DIR:-$PWD/data/gsm8k}
OUT=${OUT:-$PWD/results/$RUN_NAME}
STEPS=${STEPS:-60}
SEED=${SEED:-42}
mkdir -p "$OUT"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export WANDB_MODE=offline WANDB_DIR=$OUT
export VERL_FILE_LOGGER_PATH=$OUT/metrics.jsonl
export TOKENIZERS_PARALLELISM=false
python3 data_prep/gsm8k.py --src "$RAW_DATA" --out "$DATA_DIR" 2>&1 | tee "$OUT/data_prep.log"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed=$SEED \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.data_loader_seed=$SEED \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
  reward.custom_reward_function.path=$PWD/reward/gsm8k_reward.py \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","file","wandb"]' \
  trainer.project_name=autorlbench \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.val_before_train=True \
  trainer.total_training_steps=$STEPS \
  trainer.rollout_data_dir=$OUT/rollouts \
  trainer.validation_data_dir=$OUT/val_generations \
  trainer.default_local_dir=$OUT/checkpoints \
  "$@" 2>&1 | tee "$OUT/train.log"
