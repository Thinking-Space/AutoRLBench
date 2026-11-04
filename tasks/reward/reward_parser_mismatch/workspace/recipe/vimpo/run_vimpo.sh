#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
if [ -n "${CONDA_ENV:-}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "CONDA_ENV is set to '${CONDA_ENV}', but conda is not available." >&2
        exit 1
    fi
    conda activate "${CONDA_ENV}"
fi

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
unset VLLM_DISABLE_COMPILE_CACHE

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
RUN_MODE="${RUN_MODE:-vimpo}" # vimpo or grpo
LOGGER="${LOGGER:-wandb}" # console or wandb
RAY_RESTART="${RAY_RESTART:-1}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-}"
case "${RUN_MODE}" in
    vimpo|grpo) ;;
    *)
        echo "RUN_MODE must be 'vimpo' or 'grpo', got '${RUN_MODE}'" >&2
        exit 1
        ;;
esac

PROJECT_NAME="${PROJECT_NAME:-vimpo}"
DATA_HOME="${DATA_HOME:-${PROJECT_ROOT}/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_HOME}/math__combined_54.4k.parquet}"
VAL_AIME_FILE="${VAL_AIME_FILE:-${DATA_HOME}/math__aime_repeated_32x_960.parquet}"
VAL_MATH500_FILE="${VAL_MATH500_FILE:-${DATA_HOME}/math__math_500.parquet}"
VAL_FILES="${VAL_FILES:-['${VAL_AIME_FILE}','${VAL_MATH500_FILE}']}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-96}"
gen_prompt_bsz="${GEN_PROMPT_BSZ:-${train_prompt_bsz}}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-96}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-4096}"
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
rollout_max_model_len=$((max_prompt_length + max_response_length))

adv_estimator=grpo
norm_adv_by_std_in_grpo=True
use_kl_in_reward=False
use_kl_loss=False
kl_coef=0.0
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
ppo_epochs=1
loss_agg_mode=token-mean
use_dynamic_bsz=True
offload=False
sp_size=1
gen_tp="${GEN_TP:-1}"

vimpo_beta="${VIMPO_BETA:-0.0005}"
vimpo_actor_coeff="${VIMPO_ACTOR_COEFF:-0.005}"
vimpo_value_loss_type="${VIMPO_VALUE_LOSS_TYPE:-squared_error}"
vimpo_value_huber_delta="${VIMPO_VALUE_HUBER_DELTA:-0.005}"
vimpo_value_loss_normalized="${VIMPO_VALUE_LOSS_NORMALIZED:-False}"
vimpo_value_loss_coeff="${VIMPO_VALUE_LOSS_COEFF:-1}"
vimpo_ppo_whiten_adv="${VIMPO_PPO_WHITEN_ADV:-True}"
vimpo_adaptive_actor_coeff="${VIMPO_ADAPTIVE_ACTOR_COEFF:-False}"
vimpo_adaptive_actor_alpha="${VIMPO_ADAPTIVE_ACTOR_ALPHA:-0.025}"
vimpo_adaptive_actor_ema_decay="${VIMPO_ADAPTIVE_ACTOR_EMA_DECAY:-0.9}"
vimpo_adaptive_actor_eps="${VIMPO_ADAPTIVE_ACTOR_EPS:-1e-12}"
vimpo_cap_v0="${VIMPO_CAP_V0:-null}"
vimpo_detach_kl="${VIMPO_DETACH_KL:-True}"
vimpo_kl_estimator="${VIMPO_KL_ESTIMATOR:-exact}"
vimpo_vocab_chunk_size="${VIMPO_VOCAB_CHUNK_SIZE:-4096}"
vimpo_candidate_kl_topk="${VIMPO_CANDIDATE_KL_TOPK:-32}"
vimpo_update_ref_freq="${VIMPO_UPDATE_REF_FREQ:-0}"
value_type="${VIMPO_VALUE_TYPE:-raw}"
use_ppo_actor="${USE_PPO_ACTOR:-True}"
use_grpo_actor="${USE_GRPO_ACTOR:-False}"

if [ "${RUN_MODE}" = "vimpo" ]; then
    use_vimpo_loss=True
    exp_name="VIMPO-${MODEL_NAME}-beta${vimpo_beta}-ppoactor${use_ppo_actor}-gen${n_resp_per_prompt}-bsz${train_prompt_bsz}"
else
    use_vimpo_loss=False
    use_ppo_actor=False
    use_grpo_actor=False
    exp_name="GRPO-${MODEL_NAME}-gen${n_resp_per_prompt}-bsz${train_prompt_bsz}"
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts/${PROJECT_NAME}/${exp_name}}"
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${exp_name}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${OUTPUT_FILE}") 2>&1

echo "Run log: ${OUTPUT_FILE}"
echo "Train file: ${TRAIN_FILE}"
echo "Validation files: ${VAL_FILES}"

hydra_overrides=()
if [ -n "${ATTN_IMPLEMENTATION}" ]; then
    hydra_overrides+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}")
fi
if [ -n "${ROLLOUT_LOGPROBS_MODE}" ]; then
    hydra_overrides+=("actor_rollout_ref.rollout.logprobs_mode=${ROLLOUT_LOGPROBS_MODE}")
fi

if [ "${RAY_RESTART}" = "1" ]; then
    ray stop --force || true
    pkill -9 ray || true
    sleep 5
fi
if ! ray status >/dev/null 2>&1; then
    ray start --head --port="${RAY_PORT:-6379}" --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
fi

python3 -m recipe.vimpo.main_vimpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_p="${TOP_P:-1.0}" \
    actor_rollout_ref.rollout.top_k="${TOP_K:--1}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-0.7}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K:--1}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    vimpo_config.use_vimpo_loss=${use_vimpo_loss} \
    vimpo_config.value_type=${value_type} \
    vimpo_config.vimpo_value_loss_type=${vimpo_value_loss_type} \
    vimpo_config.vimpo_value_huber_delta=${vimpo_value_huber_delta} \
    vimpo_config.vimpo_value_loss_normalized=${vimpo_value_loss_normalized} \
    vimpo_config.vimpo_value_loss_coeff=${vimpo_value_loss_coeff} \
    vimpo_config.vimpo_ppo_whiten_adv=${vimpo_ppo_whiten_adv} \
    vimpo_config.vimpo_beta=${vimpo_beta} \
    vimpo_config.vimpo_actor_coeff=${vimpo_actor_coeff} \
    vimpo_config.vimpo_adaptive_actor_coeff=${vimpo_adaptive_actor_coeff} \
    vimpo_config.vimpo_adaptive_actor_alpha=${vimpo_adaptive_actor_alpha} \
    vimpo_config.vimpo_adaptive_actor_ema_decay=${vimpo_adaptive_actor_ema_decay} \
    vimpo_config.vimpo_adaptive_actor_eps=${vimpo_adaptive_actor_eps} \
    vimpo_config.vimpo_cap_v0=${vimpo_cap_v0} \
    vimpo_config.vimpo_detach_kl=${vimpo_detach_kl} \
    vimpo_config.vimpo_kl_estimator=${vimpo_kl_estimator} \
    vimpo_config.vocab_chunk_size=${vimpo_vocab_chunk_size} \
    vimpo_config.candidate_kl_topk=${vimpo_candidate_kl_topk} \
    vimpo_config.vimpo_update_ref_freq=${vimpo_update_ref_freq} \
    vimpo_config.use_grpo_actor=${use_grpo_actor} \
    vimpo_config.use_ppo_actor=${use_ppo_actor} \
    trainer.logger="[\"${LOGGER}\"]" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.val_before_train=True \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.save_freq="${SAVE_FREQ:-200}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "${hydra_overrides[@]}" \
    "$@"
