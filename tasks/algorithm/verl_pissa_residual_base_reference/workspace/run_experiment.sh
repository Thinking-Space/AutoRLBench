#!/usr/bin/env bash
set -euo pipefail

ROOT=${AUTORLBENCH_WORKSPACE:-/workspace}
VERL_ROOT=${VERL_ROOT:-${ROOT}/verl}
BUNDLE=${BUNDLE:-${ROOT}/assets/qwen2.5-0.5b-pissa-r8}
DATA_DIR=${DATA_DIR:-${ROOT}/data/tiny-gsm8k}
RUN_NAME=${RUN_NAME:-pissa_residual_reference}
LOG_DIR=${LOG_DIR:-${AUTORLBENCH_RESULTS:-${ROOT}/results}/experiment_logs}

RAY_JOB_TMP=${RAY_JOB_TMP:-/tmp/autorl-ray-${SLURM_JOB_ID:-local}}

mkdir -p "${LOG_DIR}" "${RAY_JOB_TMP}"
mkdir -p "${LOG_DIR}/metrics" "${LOG_DIR}/rollouts" "${LOG_DIR}/validation_generations"
export PYTHONPATH=${VERL_ROOT}:${PYTHONPATH:-}
export HF_HOME=${ROOT}/hf-cache
export RAY_TMPDIR=${RAY_JOB_TMP}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR=${LOG_DIR}
export WANDB_DISABLE_CODE=true
export VERL_FILE_LOGGER_PATH=${LOG_DIR}/metrics/all_steps.jsonl
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Skipjack exports this ROCm variable even on NVIDIA nodes. Ray propagates it
# together with CUDA_VISIBLE_DEVICES, which verl rejects as ambiguous.
unset ROCR_VISIBLE_DEVICES
export VERL_BASE_KL_DEBUG_DIR=${LOG_DIR}/base_kl_debug
export VERL_ACTOR_POLICY_DEBUG_DIR=${LOG_DIR}/rollout_policy_debug
export VERL_ROLLOUT_POLICY_DEBUG_DIR=${LOG_DIR}/rollout_policy_debug

printf '%s\n' 'packaged-defective-snapshot' > "${LOG_DIR}/source_commit.txt"
python -m pip freeze > "${LOG_DIR}/environment-freeze.txt"
nvidia-smi -q > "${LOG_DIR}/nvidia-smi.txt"

cd "${VERL_ROOT}"
python -m verl.trainer.main_ppo \
  ray_kwargs.ray_init.num_cpus=10 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/validation.parquet" \
  data.train_batch_size=4 \
  data.max_prompt_length=256 \
  data.max_response_length=32 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed=2026 \
  actor_rollout_ref.model.path="${BUNDLE}/residual_base" \
  actor_rollout_ref.model.lora_adapter_path="${BUNDLE}/pissa_adapter" \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=8 \
  +actor_rollout_ref.model.use_dora=False \
  +actor_rollout_ref.model.lora_residual_base=True \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.05 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  reward_model.reward_manager=naive \
  trainer.logger='[console,wandb,file]' \
  trainer.project_name=AutoRLBench-Minimal-RL \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=4 \
  trainer.total_training_steps=4 \
  trainer.default_local_dir="${LOG_DIR}/checkpoints-disabled" \
  trainer.rollout_data_dir="${LOG_DIR}/rollouts" \
  trainer.validation_data_dir="${LOG_DIR}/validation_generations" \
  trainer.log_val_generations=4 \
  +trainer.log_base_model_kl=True \
  +trainer.log_final_policy_health=True \
  2>&1 | tee "${LOG_DIR}/train.log"
