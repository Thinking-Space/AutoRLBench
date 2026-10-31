#!/usr/bin/env bash
# GRPO on GSM8K with a calculator tool: Qwen3-1.7B (thinking mode), verl 0.9.0 agent loop,
# FSDP actor + vLLM async rollout (colocated), 1 GPU, 100 steps, held-out eval every 10 steps
# on 512 GSM8K test problems and 512 GSM-Hard problems (same questions, large numbers).
#
#   bash run_calc_grpo.sh                  # logs go to $LOG_DIR (default results/run)
#
# Env knobs: LOG_DIR, MODEL_PATH, DATA_DIR, N_GPUS, TOTAL_STEPS, TEST_FREQ, EXPERIMENT_NAME.
# trainer.use_v1=False selects the ray_trainer.py loop; the v1 trainer needs the transfer_queue
# package, which this environment does not ship.
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=${MODEL_PATH:-$W/models/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-$W/data/gsm8k}
RAW_DIR=${RAW_DIR:-$W/data/raw/gsm8k}
RAW_HARD_DIR=${RAW_HARD_DIR:-$W/data/raw/gsm-hard}
LOG_DIR=${LOG_DIR:-$W/results/run}
N_GPUS=${N_GPUS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-10}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_gsm8k_calc_grpo}

for need in "$MODEL_PATH/config.json" "$RAW_DIR/main/train-00000-of-00001.parquet" "$RAW_HARD_DIR/gsmhardv2.jsonl"; do
    [ -e "$need" ] || { echo "missing $need (run environment/prepare.sh first)" >&2; exit 1; }
done
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ] || [ ! -f "$DATA_DIR/test_hard.parquet" ]; then
    python3 "$W/calc_agent/prepare_gsm8k.py" --raw "$RAW_DIR" --raw-hard "$RAW_HARD_DIR" --out "$DATA_DIR"
fi

mkdir -p "$LOG_DIR"
export PYTHONPATH="$W${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
export RAY_ADDRESS=${RAY_ADDRESS:-local} RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_usage_threshold=0.99
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=${WANDB_DIR:-$LOG_DIR}
export VERL_FILE_LOGGER_PATH="$LOG_DIR/metrics.jsonl"
export CALC_AGENT_DUMP_DIR="$LOG_DIR/episodes"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="[$DATA_DIR/test.parquet,$DATA_DIR/test_hard.parquet]" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.seed=2026 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.function_tool_path="$W/calc_agent/calculator.py" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$W/calc_agent/agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=calc_agent \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name=gsm8k_calc_agent \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$LOG_DIR/checkpoints" \
    trainer.rollout_data_dir="$LOG_DIR/rollout_data" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="$TEST_FREQ" \
    trainer.val_before_train=True \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "$LOG_DIR/train.log"
exit "${PIPESTATUS[0]}"
