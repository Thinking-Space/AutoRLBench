#!/bin/bash
# GRPO (verl FSDP + vLLM) on an integer-answer math pool (Skywork-OR1 / POLARIS / DAPO-Math-17k)
# from Qwen/Qwen3-4B. Builds the RL parquet with scripts/make_rl_data.py, then trains.
# Env vars configure the run; extra Hydra overrides on the command line are appended last.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
export WS
MODEL=${MODEL:-$WS/models/Qwen3-4B}
NGPU=${NGPU:-2}
EXP=${EXP:-grpo}
RESP=${RESP:-8192}
BS=${BS:-32}
MINI=${MINI:-16}
N=${N:-8}
LR=${LR:-1e-6}
STEPS=${STEPS:-5}
GPU_UTIL=${GPU_UTIL:-0.5}
SEED=${SEED:-1}
MAXTOK=$(( RESP + 1024 ))
LOG_DIR=${LOG_DIR:-$WS/outputs/$EXP}
mkdir -p "$LOG_DIR"
export PYTHONPATH="$WS/verl:$WS/third_party/python${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$LOG_DIR"
export RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=false

python3 "$WS/scripts/make_rl_data.py" > "$LOG_DIR/make_rl_data.log"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$WS/data/rl_train.parquet" \
  data.val_files="$WS/data/rl_val.parquet" \
  data.train_batch_size=$BS \
  data.max_prompt_length=1024 \
  data.max_response_length=$RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.seed=$SEED \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD:-False} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD:-False} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.max_model_len=$MAXTOK \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  reward.custom_reward_function.path="$WS/scripts/math_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.logger='["console","file","wandb"]' \
  trainer.project_name=math \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$LOG_DIR/checkpoints" \
  trainer.rollout_data_dir="$LOG_DIR/rollouts" \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_training_steps=$STEPS \
  trainer.total_epochs=10 \
  "$@" 2>&1 | tee "$LOG_DIR/train.log"
