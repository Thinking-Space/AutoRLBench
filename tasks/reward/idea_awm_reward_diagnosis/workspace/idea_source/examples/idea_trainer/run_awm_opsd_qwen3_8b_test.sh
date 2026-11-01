#!/usr/bin/env bash
set -euo pipefail

# The hosted judge must connect directly; inherited proxy variables on
# training nodes can cause intermittent connection errors and read timeouts.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-seed}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || { echo "conda is required to activate $CONDA_ENV" >&2; exit 1; }
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

MODEL_PATH="${MODEL_PATH:-/local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_20260914_100457/global_step_693}"
AWM_ROOT="${AWM_ROOT:-/workspace/awm_source}"
AWM_DATA_ROOT="${AWM_DATA_ROOT:-/workspace/data/AgentWorldModel-1K}"
AWM_PYTHON="${AWM_PYTHON:-$AWM_ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$AWM_DATA_ROOT/verl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/awm_opsd_qwen3_8b_step693_wandb_v2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/local/user/models/opsd-tool/awm_opsd_qwen3_8b_step693_wandb_v2}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
GROUP_SIZE="${GROUP_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
AWM_REWARD_MODE="${AWM_REWARD_MODE:-code}"
# Hosted service settings must be supplied by the user, never bundled secrets.
# The recorded code-reward path does not require these credentials.
AWM_JUDGE_BASE_URL="${AWM_JUDGE_BASE_URL:-}"
AWM_JUDGE_API_KEY="${AWM_JUDGE_API_KEY:-EMPTY}"
AWM_JUDGE_MODEL="${AWM_JUDGE_MODEL:-}"
AWM_JUDGE_CONCURRENCY="${AWM_JUDGE_CONCURRENCY:-20}"
AWM_JUDGE_TIMEOUT="${AWM_JUDGE_TIMEOUT:-120}"
AWM_JUDGE_MAX_RETRIES="${AWM_JUDGE_MAX_RETRIES:-3}"
AWM_STARTUP_CONCURRENCY="${AWM_STARTUP_CONCURRENCY:-32}"
case "$AWM_REWARD_MODE" in
    code) VERIFIER_FILE="$AWM_DATA_ROOT/gen_verifier.pure_code.jsonl" ;;
    sql) VERIFIER_FILE="$AWM_DATA_ROOT/gen_verifier.jsonl" ;;
    *) echo "AWM_REWARD_MODE must be code or sql" >&2; exit 1 ;;
esac

for path in "$MODEL_PATH/config.json" "$AWM_DATA_ROOT/gen_tasks.jsonl" \
    "$AWM_DATA_ROOT/gen_envs.jsonl" "$AWM_DATA_ROOT/gen_db.jsonl" \
    "$AWM_DATA_ROOT/gen_sample.jsonl" "$VERIFIER_FILE" \
    "$AWM_PYTHON"; do
    [[ -e "$path" ]] || { echo "Required path not found: $path" >&2; exit 1; }
done

if (( TRAIN_BATCH_SIZE * GROUP_SIZE % N_GPUS_PER_NODE != 0 )); then
    echo "TRAIN_BATCH_SIZE * GROUP_SIZE must be divisible by N_GPUS_PER_NODE" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
if [[ "${DRY_RUN:-false}" != "true" && (! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet") ]]; then
    python3 examples/data_preprocess/preprocess_awm_dataset.py \
        --data-root "$AWM_DATA_ROOT" --local-dir "$DATA_DIR" --val-size 128
fi

# The LLM judge is a hosted GLM API and does not reserve local GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export TOKENIZERS_PARALLELISM=false
# Optional W&B controls: WANDB_ENTITY, WANDB_PROJECT, WANDB_NAME,
# WANDB_MODE=online|offline|disabled, and WANDB_API_KEY.
export WANDB_PROJECT="${WANDB_PROJECT:-agent_world_model}"
export WANDB_NAME="${WANDB_NAME:-opsd_qwen3_8b_step715_skill_teacher}"
export RAY_TMPDIR="${RAY_TMPDIR:-/local/user/ray}"
mkdir -p "$RAY_TMPDIR"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
ulimit -u 65536

launch_args=(
    algorithm.adv_estimator=seed
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.val_batch_size="$VAL_BATCH_SIZE"
    data.max_prompt_length=12288
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=left
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.opd_loss_coef="${SEED_OPD_LOSS_COEF:-0.01}"
    actor_rollout_ref.actor.opd_gate_beta="${SEED_OPD_GATE_BETA:-5.0}"
    actor_rollout_ref.actor.skill_gen_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.max_model_len=14336
    actor_rollout_ref.rollout.max_num_batched_tokens=14336
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    # AWM emits the complete format-failure reward (-1.0) and terminates the
    # episode in the environment.  Do not apply verl's separate invalid-action
    # penalty, which would double-count that reward.
    actor_rollout_ref.actor.use_invalid_action_penalty=False
    algorithm.use_kl_in_reward=False
    algorithm.gamma=0.95
    algorithm.seed.mode=mean_std_norm
    algorithm.seed.step_advantage_w=0.0
    algorithm.seed.episode_skill_teacher_advantage_w=0.0
    algorithm.seed.step_skill_teacher_advantage_w=0.0
    algorithm.seed.skill_mode=episode_only
    algorithm.seed.skill_teacher_mode=step_priority
    algorithm.seed.enable_analysis=True
    algorithm.seed.selector=llm
    algorithm.seed.analysis_backend=policy_vllm
    algorithm.seed.analysis_context_length=12288
    algorithm.seed.analysis_max_completion_tokens=2048
    algorithm.seed.analysis_max_step_skills_per_traj=0
    algorithm.seed.analysis_prompt_version=seed
    algorithm.seed.analysis_include_episode_summary=True
    algorithm.seed.analysis_dump_dir="$OUTPUT_DIR/idea_analysis"
    algorithm.seed.augmented_observation_dump_dir="$OUTPUT_DIR/seed_augmented_observations"
    algorithm.seed.state_group_dump_dir="$OUTPUT_DIR/seed_state_group"
    algorithm.seed.failed_only=False
    algorithm.seed.normalize_teacher_adv=False
    algorithm.seed.skill_gen.max_samples=all
    env.env_name=agent_world_model
    env.seed=0
    env.max_steps="$MAX_STEPS"
    env.history_length="$MAX_STEPS"
    env.rollout.n="$GROUP_SIZE"
    env.awm.root="$AWM_ROOT"
    env.awm.python="$AWM_PYTHON"
    env.awm.data_root="$AWM_DATA_ROOT"
    env.awm.work_root="${AWM_WORK_ROOT:-$OUTPUT_DIR/awm_runtime}"
    env.awm.tool_timeout="${AWM_TOOL_TIMEOUT:-60}"
    env.awm.reward_mode="$AWM_REWARD_MODE"
    env.awm.judge_base_url="$AWM_JUDGE_BASE_URL"
    env.awm.judge_api_key="$AWM_JUDGE_API_KEY"
    env.awm.judge_model="$AWM_JUDGE_MODEL"
    +env.awm.judge_concurrency="$AWM_JUDGE_CONCURRENCY"
    env.awm.judge_timeout="$AWM_JUDGE_TIMEOUT"
    env.awm.judge_max_retries="$AWM_JUDGE_MAX_RETRIES"
    +env.awm.startup_concurrency="$AWM_STARTUP_CONCURRENCY"
    trainer.critic_warmup=0
    # Enable W&B by default while retaining console output. Override with
    # TRAINER_LOGGER="['console']" to disable remote tracking.
    trainer.logger="${TRAINER_LOGGER:-['console']}"
    trainer.project_name=agent_world_model
    trainer.experiment_name=opsd_qwen3_8b_step715_skill_teacher
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ:-10}"
    trainer.test_freq="${TEST_FREQ:-50}"
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
    trainer.val_before_train=False
    trainer.default_local_dir="$CHECKPOINT_DIR"
    trainer.rollout_data_dir="$OUTPUT_DIR/rollouts"
    ray_init.num_cpus="${RAY_NUM_CPUS:-64}"
)

echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Effective rollout batch: $((TRAIN_BATCH_SIZE * GROUP_SIZE)); GPUs: $N_GPUS_PER_NODE; reward mode: $AWM_REWARD_MODE"
if [[ "$AWM_REWARD_MODE" == "sql" ]]; then
    echo "Judge: $AWM_JUDGE_BASE_URL (model: $AWM_JUDGE_MODEL)"
    echo "Judge max concurrency: $AWM_JUDGE_CONCURRENCY (hard cap: 10)"
    echo "Judge timeout/retries: ${AWM_JUDGE_TIMEOUT}s / ${AWM_JUDGE_MAX_RETRIES}"
    echo "AWM startup concurrency: $AWM_STARTUP_CONCURRENCY"
    echo "Judge trace: $OUTPUT_DIR/awm_runtime/llm_judge.jsonl"
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf 'python3 -m verl.trainer.main_ppo'
    printf ' %q' "${launch_args[@]}" "$@"
    printf '\n'
    exit 0
fi

# Ray workers can outlive the launcher after an interrupt.  Keep a final
# cleanup hook in the launcher so MCP/uvicorn processes are not left behind.
TRAIN_PID=""
cleanup_awm_processes() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
    fi
    local base="${AWM_WORK_ROOT:-$OUTPUT_DIR/awm_runtime}"
    if [[ -d "$base" ]] && command -v lsof >/dev/null 2>&1; then
        local pids
        pids=$(lsof +D "$base" 2>/dev/null | awk 'NR > 1 && $2 ~ /^[0-9]+$/ {print $2}' | sort -nu)
        if [[ -n "$pids" ]]; then
            echo "清理 AWM/MCP 残留进程: $(echo "$pids" | wc -w)"
            echo "$pids" | xargs -r kill -TERM 2>/dev/null || true
            sleep 1
            echo "$pids" | xargs -r kill -KILL 2>/dev/null || true
        fi
    fi
    exit "$status"
}
trap cleanup_awm_processes EXIT INT TERM

python3 -m verl.trainer.main_ppo "${launch_args[@]}" "$@" &
TRAIN_PID=$!
wait "$TRAIN_PID"
