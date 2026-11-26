#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export QAT_NEMOTRON_NATIVE_LOADER=1
export PYTHONUNBUFFERED=1
log_dir=$(realpath -m "${LOG_DIR:-experiment_logs}")
mkdir -p "$log_dir"
exec python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    critic.enable=false reward_model.enable=false reward_model.reward_manager=naive \
    custom_reward_function.path=math_reward.py custom_reward_function.name=score \
    data.train_files=assets/split/train.parquet data.val_files=assets/split/validation.parquet \
    data.max_prompt_length=512 data.max_response_length=16384 data.train_batch_size=8 \
    data.shuffle=false data.seed=42 data.truncation=error \
    actor_rollout_ref.model.path=assets/model actor_rollout_ref.model.trust_remote_code=true \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["NemotronHBlock"]' \
    actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=0.0000015 \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=4 +actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=32 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    trainer.logger='["console"]' trainer.project_name=aime-math trainer.experiment_name=nemotron-grpo \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=1 trainer.total_training_steps=2 \
    trainer.val_before_train=true trainer.test_freq=1 trainer.save_freq=-1 trainer.resume_mode=disable \
    trainer.rollout_dump.enable=true trainer.rollout_dump.fail_on_error=true \
    "trainer.rollout_dump.dir=$log_dir/rollouts" "trainer.validation_data_dir=$log_dir/validation" \
    "trainer.default_local_dir=$log_dir/checkpoints" "$@"
