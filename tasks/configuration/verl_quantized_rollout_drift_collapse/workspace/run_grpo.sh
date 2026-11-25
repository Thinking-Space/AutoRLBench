#!/bin/bash
# GRPO (verl FSDP + vLLM) on Countdown with a throughput-tuned rollout engine:
# the vLLM engine runs online fp8-quantized weights while the trainer stays bf16.
# Builds the RL parquet with scripts/data_prep_countdown.py, then trains.
# Env vars configure the run; extra Hydra overrides on the command line are appended last.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
export WS

MODEL=${MODEL:-$WS/models/Qwen2.5-1.5B-Instruct} # a small instruct model that follows the <answer> format on Countdown
NGPU=${NGPU:-2}
EXP=${EXP:-grpo}
RESP=${RESP:-1024}
BS=${BS:-64}
MINI=${MINI:-32}
N=${N:-8}
LR=${LR:-2e-6}
STEPS=${STEPS:-50}
GPU_UTIL=${GPU_UTIL:-0.35}
SEED=${SEED:-1}
QUANT=${QUANT-fp8}         # rollout weight quantization; empty = off
KV_DTYPE=${KV_DTYPE:-auto} # rollout KV-cache dtype; fp8 = on
RAW_DATA=${RAW_DATA:-$WS/raw/countdown}
DATA_DIR=${DATA_DIR:-$WS/data/countdown}
MAXTOK=$(( RESP + 512 ))
LOG_DIR=${LOG_DIR:-$WS/outputs/$EXP}
mkdir -p "$LOG_DIR"
export PYTHONPATH="$WS/verl${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$LOG_DIR" WANDB_DISABLE_CODE=true
export RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=false
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-$LOG_DIR/metrics.jsonl}

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    test -d "$RAW_DATA" || { echo "countdown snapshot missing: $RAW_DATA (fetch Jiayi-Pan/Countdown-Tasks-3to4 first)"; exit 1; }
    python3 "$WS/scripts/data_prep_countdown.py" --src "$RAW_DATA" --out "$DATA_DIR" 2>&1 | tee "$LOG_DIR/data_prep.log"
fi

ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size=$BS
    data.max_prompt_length=256
    data.max_response_length=$RESP
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.seed=$SEED
    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.optim.weight_decay=0.0
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXTOK
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAXTOK
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL
    actor_rollout_ref.rollout.n=$N
    actor_rollout_ref.rollout.max_model_len=$MAXTOK
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAXTOK
    actor_rollout_ref.rollout.agent.num_workers=8
    # keep the rollout engine resident across training steps (stable memory
    # profile; the vLLM sleep/wake cycle is not exercised)
    actor_rollout_ref.rollout.free_cache_engine=False
    # ship the whole model in a single weight-sync bucket so per-layer fused
    # (q/k/v, gate/up) tensors never straddle a bucket boundary
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=16384
    reward.custom_reward_function.path="$WS/scripts/countdown_reward.py"
    reward.custom_reward_function.name=compute_score
    trainer.logger='["console","file","wandb"]'
    trainer.project_name=countdown_grpo
    trainer.experiment_name="$EXP"
    trainer.default_local_dir="$LOG_DIR/checkpoints"
    trainer.rollout_data_dir="$LOG_DIR/rollouts"
    trainer.n_gpus_per_node=$NGPU
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_training_steps=$STEPS
    trainer.total_epochs=10
)
if [ -n "$QUANT" ]; then
    ARGS+=(actor_rollout_ref.rollout.quantization="$QUANT")
fi
if [ "$KV_DTYPE" != "auto" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=$KV_DTYPE")
fi

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee "$LOG_DIR/train.log"
