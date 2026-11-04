#!/usr/bin/env bash
# GRPO | Qwen2.5-0.5B-Instruct | GSM8K train 7,473 / GSM8K test 1,319 held-out | FSDP + vLLM | 1 GPU
#
# verl GSM8K GRPO quickstart recipe, scaled to one GPU, with the MATH-style prompt from
# examples/data_preprocess/gsm8k_boxed.py ('... output the final answer within \\boxed{}.'). The reward
# is Math-Verify through verl's built-in scorer (rewards/math_verify_reward.py, wired in with
# reward.custom_reward_function); verl uses the same function for the held-out metric
# val-core/openai/gsm8k/acc/mean@1 on the full GSM8K test split (greedy).
#
# trainer.use_v1=False selects the verl/trainer/ppo/ray_trainer.py loop; the v1 trainer needs
# the optional TransferQueue package, which is not installed.
#
# Everything the run writes goes to $LOG_DIR:
#   train.log                       full stdout/stderr of the trainer
#   metrics.jsonl                   one JSON line per training step (verl file logger)
#   wandb/                          offline W&B run
#   rollouts/<step>.jsonl           training rollouts (prompt, response, score) per step
#   val_generations/<step>.jsonl    held-out generations at each evaluation
#   rollouts/ and val_generations/ carry the score the reward assigned to each response.

set -xeuo pipefail

W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$W"

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-$W/models/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$W/data/gsm8k_boxed/train.parquet}
VAL_FILE=${VAL_FILE:-$W/data/gsm8k_boxed/test.parquet}
LOG_DIR=${LOG_DIR:-$W/outputs/run}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

TOTAL_STEPS=${TOTAL_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-10}
SEED=${SEED:-42}

TRAIN_BATCH_SIZE=128         # prompts per step
ROLLOUT_N=8                  # responses per prompt (GRPO group size)
PPO_MINI_BATCH_SIZE=32       # 4 optimizer steps per training step
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512

ACTOR_LR=2e-6
KL_LOSS_COEF=0.001

PROJECT_NAME=${PROJECT_NAME:-grpo_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_0_5b_gsm8k_boxed_grpo}
# ---- end user-adjustable ----

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "model not found at $MODEL_PATH" >&2; exit 1
fi
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    python3 examples/data_preprocess/gsm8k_boxed.py --local_dataset_path "$W/data/raw/gsm8k" --local_save_dir "$(dirname "$TRAIN_FILE")"
fi

mkdir -p "$LOG_DIR"
export PYTHONPATH="$W${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$LOG_DIR}
export VERL_FILE_LOGGER_PATH="$LOG_DIR/metrics.jsonl"
export TOKENIZERS_PARALLELISM=false

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation=error
    data.seed=$SEED
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.data_loader_seed=$SEED
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.seed=$SEED
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576
)

REWARD=(
    reward.custom_reward_function.path="$W/rewards/math_verify_reward.py"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.use_v1=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb","file"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    trainer.n_gpus_per_node=$NGPUS_PER_NODE
    trainer.nnodes=1
    trainer.val_before_train=True
    trainer.test_freq=$TEST_FREQ
    trainer.save_freq=-1
    trainer.total_epochs=100
    trainer.total_training_steps=$TOTAL_STEPS
    trainer.rollout_data_dir="$LOG_DIR/rollouts"
    trainer.validation_data_dir="$LOG_DIR/val_generations"
    trainer.default_local_dir="$LOG_DIR/checkpoints"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "$LOG_DIR/train.log"
