#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-/local/user/Qwen/Qwen3-8B}"
TRAIN="${TRAIN:-$PROJECT_ROOT/outputs/awm_qwen38_sft_data/sft_episode_skill_train.parquet}"
VAL="${VAL:-$PROJECT_ROOT/outputs/awm_qwen38_sft_data/sft_episode_skill_val.parquet}"
TURN_ROLLOUTS="${TURN_ROLLOUTS:-$PROJECT_ROOT/outputs/awm_qwen38_27b_rollouts_20pct_20260909/sft_rollouts_turn_level.jsonl}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/qwen3_8b_awm_skill_sft_verl_mixed_data}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-null}"
SAVE_FREQ_STEPS="${SAVE_FREQ_STEPS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

cd "$PROJECT_ROOT"
# The node's default NCCL transport can deadlock during FSDP startup; use the
# socket transport and fail fast on collective errors.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
exec torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  "data.train_files=$TRAIN" \
  "data.val_files=$VAL" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" \
  "data.prompt_key=prompt" \
  "data.response_key=response" \
  "data.prompt_dict_keys=[]" \
  "data.response_dict_keys=[]" \
  "data.max_length=$MAX_LENGTH" \
  "data.truncation=left" \
  "data.multiturn.enable=False" \
  "data.chat_template=null" \
  "+data.apply_chat_template_kwargs.enable_thinking=False" \
  "model.partial_pretrain=$MODEL" \
  "model.trust_remote_code=True" \
  "model.strategy=fsdp" \
  "+model.torch_dtype=bfloat16" \
  "model.enable_gradient_checkpointing=True" \
  "model.fsdp_config.wrap_policy.min_num_params=0" \
  "model.fsdp_config.cpu_offload=False" \
  "optim.lr=5e-6" \
  "optim.betas=[0.9,0.95]" \
  "optim.weight_decay=0.01" \
  "optim.warmup_steps_ratio=0.1" \
  "optim.lr_scheduler=cosine" \
  "trainer.default_local_dir=$OUTPUT" \
  "trainer.default_hdfs_dir=null" \
  "trainer.project_name=awm-qwen3-8b-sft" \
  "trainer.experiment_name=qwen3-8b-awm-skill-sft" \
  "trainer.logger=['console']" \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  "trainer.save_freq_steps=$SAVE_FREQ_STEPS" \
  "ulysses_sequence_parallel_size=1" \
  "use_remove_padding=False" \
  "$@"
