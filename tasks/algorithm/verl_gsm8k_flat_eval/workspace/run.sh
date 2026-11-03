#!/usr/bin/env bash
# Re-run the recorded single-GPU, 50-step VERL / GRPO experiment.
set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$WORKSPACE_ROOT/verl"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/assets/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE_ROOT/assets/gsm8k/train.parquet}"
VAL_SOURCE="${VAL_SOURCE:-$WORKSPACE_ROOT/assets/gsm8k/test.parquet}"
RESULTS_ROOT="${AUTORLBENCH_RESULTS:-$WORKSPACE_ROOT/results}"
OUTPUT_BASE="${OUTPUT_BASE:-$RESULTS_ROOT/experiment_logs}"
SEED="${SEED:-123}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SAVE_FREQ="${SAVE_FREQ:-50}"

[[ -s "$MODEL_PATH/config.json" ]] || { echo "ERROR: model config missing: $MODEL_PATH/config.json" >&2; exit 2; }
[[ -s "$TRAIN_FILE" && -s "$VAL_SOURCE" ]] || { echo 'ERROR: VERL-formatted GSM8K train/test parquet is missing.' >&2; exit 2; }
[[ "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: EVAL_LIMIT must be a positive integer.' >&2; exit 2; }

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${RUN_DIR:-$OUTPUT_BASE}"
mkdir -p "$RUN_DIR"
[[ ! -e "$RUN_DIR/train.log" ]] || { echo "ERROR: log exists; choose a new RUN_DIR: $RUN_DIR" >&2; exit 2; }
VAL_FILE="$RUN_DIR/validation_${EVAL_LIMIT}.parquet"

"$PYTHON_BIN" - "$VAL_SOURCE" "$VAL_FILE" "$EVAL_LIMIT" <<'PY'
import sys
from datasets import load_dataset
source, destination, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = load_dataset("parquet", data_files=source, split="train")
count = min(limit, len(data))
data.select(range(count)).to_parquet(destination)
print(f"validation_subset={destination} rows={count} source_rows={len(data)}")
PY

mkdir -p "$RUN_DIR/wandb"
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_PROJECT=verl_gsm8k
export WANDB_NAME="run_${SEED}"
export WANDB_RUN_GROUP=gsm8k_repro
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$VERL_ROOT"
echo "model=$MODEL_PATH train=$TRAIN_FILE validation=$VAL_FILE"
echo "run=$RUN_DIR wandb_mode=$WANDB_MODE wandb_dir=$WANDB_DIR"
echo "steps=$TRAIN_STEPS eval_every=$EVAL_EVERY eval_rows=$EVAL_LIMIT rollout.n=8 batch=1 response_cap=$MAX_RESPONSE_LENGTH model_context=$MAX_MODEL_LEN save_freq=$SAVE_FREQ"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  "data.train_files=[$TRAIN_FILE]" \
  "data.val_files=[$VAL_FILE]" \
  data.train_batch_size=1 \
  data.shuffle=True \
  "+data.seed=$SEED" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref.actor.checkpoint.contents=[model,hf_model] \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  "+actor_rollout_ref.rollout.seed=$SEED" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.log_val_generations=8 \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  "trainer.save_freq=$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep=1 \
  "trainer.test_freq=$EVAL_EVERY" \
  trainer.project_name=verl_gsm8k \
  trainer.experiment_name=run_${SEED} \
  "trainer.default_local_dir=$RUN_DIR/checkpoints" \
  trainer.total_epochs=1 \
  "trainer.total_training_steps=$TRAIN_STEPS" \
  trainer.resume_mode=disable \
  ray_init.num_cpus=4 \
  2>&1 | tee "$RUN_DIR/train.log"

echo "RUN_DIR=$RUN_DIR"
