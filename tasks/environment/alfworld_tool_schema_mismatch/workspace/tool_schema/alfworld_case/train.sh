#!/usr/bin/env bash
set -euo pipefail

# Run inside the runtime container with tool_schema mounted at /workspace/tool_schema.
CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PYTHONPATH="$CASE_ROOT:$CASE_ROOT/vendor/Agent-R1${PYTHONPATH:+:$PYTHONPATH}"
export ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-$CASE_ROOT/datasets/alfworld/balanced1008-s42}"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
RUN_DIR="${RUN_DIR:-$CASE_ROOT/runs/alfworld-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR"
cd "$CASE_ROOT/vendor/Agent-R1"
AGENT_WORKERS=16
# Reserve CPU slots outside FSDP placement groups for servers and environment workers.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((8 + 4 * ${N_GPUS:-1} + AGENT_WORKERS))}"
python -m agent_r1.trainer.main_agent_ppo \
    hydra.run.dir="$RUN_DIR/hydra" \
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS" \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    +ray_kwargs.ray_init._temp_dir="$RUN_DIR/ray" \
    algorithm.adv_estimator=grpo \
    data.train_files="$CASE_ROOT/datasets/alfworld/difficulty-v1-s42/train.parquet" \
    data.val_files="$ALFWORLD_DATA_ROOT/validation.parquet" \
    data.train_batch_size=16 data.val_batch_size=8 data.seed=42 data.shuffle=False \
    data.max_prompt_length=4096 data.max_response_length=12288 \
    data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH:-$CASE_ROOT/Qwen3-4B}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=16384 actor_rollout_ref.rollout.max_num_seqs=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.agent.default_agent_flow=alfworld_schema \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CASE_ROOT/alfworld_case/alfworld_schema.yaml" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    reward_model.enable=False critic.enable=False algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$CASE_ROOT/vendor/Agent-R1/recipes/alfworld/reward_fn.py" \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=alfworld-tool-schema trainer.experiment_name=alfworld \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.validation_data_dir="$RUN_DIR/validation" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    trainer.n_gpus_per_node="${N_GPUS:-1}" trainer.nnodes=1 \
    trainer.val_before_train=True trainer.test_freq=20 trainer.save_freq=-1 \
    trainer.total_training_steps=63 trainer.total_epochs=1 "$@" 2>&1 | tee "$RUN_DIR/train.log"
