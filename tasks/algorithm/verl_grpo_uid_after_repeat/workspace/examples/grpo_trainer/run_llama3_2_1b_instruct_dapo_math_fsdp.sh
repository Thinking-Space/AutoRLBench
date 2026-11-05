#!/usr/bin/env bash
# GRPO | Llama-3.2-1B-Instruct (dense) | FSDP + vLLM | DAPO-Math-17k
#
# Dense (LlamaForCausalLM), NOT MoE.
#
# Reward / advantage (defaults):
#   - algorithm.adv_estimator = grpo
#   - reward.reward_manager.name = dapo (overlong off by default)
#   - score: examples/grpo_trainer/reward_math_dapo_01.py::compute_score
#     (0/1: correct 1.0 / incorrect 0.0; same extract as math_dapo)
#   - trainer.resume_mode = disable  (always train from scratch unless overridden)
#   - DAPO filter_groups / dynamic sampling is NOT enabled (default null)

set -euo pipefail

# ---- logging / wandb ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${VERL_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/llama3_2_1b_instruct_dapo_math_fsdp_${TIMESTAMP}.log"
# Offline wandb: keep all run data local, no API key / network required.
# Logs land under ${WANDB_DIR}/wandb/ (sync later with `wandb sync` if desired).
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_dapo_math}"
# Per-step metrics as JSONL, written unbuffered by verl's `file` tracking
# backend. The `console` backend prints the same numbers from inside the
# TaskRunner Ray actor, and whether Ray forwards that actor's stdout to the
# driver (and hence into LOG_FILE) varies with the Ray build, so the text log
# is not a dependable record of the metrics. This file always is.
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${LOG_DIR}/metrics_${TIMESTAMP}.jsonl}"
# ---- end logging / wandb ----

# Resolve model path for host / docker.
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /models/Llama-3.2-1B-instruct ]]; then
        MODEL_PATH=/models/Llama-3.2-1B-instruct
    elif [[ -d /workspace/verl/Llama-3.2-1B-instruct ]]; then
        MODEL_PATH=/workspace/verl/Llama-3.2-1B-instruct
    else
        MODEL_PATH="${VERL_ROOT}/verl/Llama-3.2-1B-instruct"
    fi
fi

# Dataset: DAPO-Math-17k (verl parquet).
if [[ -z "${TRAIN_FILE:-}" ]]; then
    if [[ -f /workspace/DAPO-Math-17k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/DAPO-Math-17k_verl/train.parquet
    elif [[ -f /workspace/verl/datasets/DAPO-Math-17k_verl/train.parquet ]]; then
        TRAIN_FILE=/workspace/verl/datasets/DAPO-Math-17k_verl/train.parquet
    else
        TRAIN_FILE=/datasets/DAPO-Math-17k_verl/train.parquet
    fi
fi
if [[ -z "${TEST_FILE:-}" ]]; then
    if [[ -f /workspace/DAPO-Math-17k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/DAPO-Math-17k_verl/test.parquet
    elif [[ -f /workspace/verl/datasets/DAPO-Math-17k_verl/test.parquet ]]; then
        TEST_FILE=/workspace/verl/datasets/DAPO-Math-17k_verl/test.parquet
    else
        TEST_FILE=/datasets/DAPO-Math-17k_verl/test.parquet
    fi
fi

NNODES=${NNODES:-1}
# 1B dense fits easily; 2 GPUs.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

# 8 prompts x ROLLOUT_N=8 samples = 64 sequences per update, so GRPO groups
# mix correct and incorrect responses often enough to produce gradient at the
# ~3-10% base accuracy of this model on DAPO-Math.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
# KL loss off by default for this dense DAPO-Math run.
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Dense 1B: TP=1 (one full copy per rollout engine; with 2 GPUs -> DP=2)
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-8}
# Llama-3.2 HF config has max_position_embeddings=131072; MUST cap for vLLM.
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}

# DAPO overlong buffer (disabled by default; must be < max_response_length when enabled)
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}
# No checkpoints: this recipe is scored from logged metrics only.
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}

USE_V1=${USE_V1:-False}
# "auto" would resume from checkpoints/.../latest; default disable = from-scratch.
RESUME_MODE=${RESUME_MODE:-disable}
# 0/1 reward (vs math_dapo native +1/-1)
REWARD_FN_PATH=${REWARD_FN_PATH:-${SCRIPT_DIR}/reward_math_dapo_01.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers=0
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path=${REWARD_FN_PATH}
    reward.custom_reward_function.name=${REWARD_FN_NAME}
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    # file: per-step JSONL at VERL_FILE_LOGGER_PATH, the authoritative record.
    # console: same numbers to the TaskRunner's stdout (only reaches LOG_FILE
    #   when the Ray build forwards actor stdout to the driver).
    # wandb: offline run under WANDB_DIR/wandb.
    trainer.logger='["file","console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.use_v1=${USE_V1}
    trainer.resume_mode=${RESUME_MODE}
)

EXTRA=(
)

########################### launch ###########################
echo "Logging to ${LOG_FILE} (also echoed to the console)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE} TEST_FILE=${TEST_FILE}"
echo "adv=grpo reward_manager=dapo score=0/1(${REWARD_FN_PATH}) resume_mode=${RESUME_MODE} max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
