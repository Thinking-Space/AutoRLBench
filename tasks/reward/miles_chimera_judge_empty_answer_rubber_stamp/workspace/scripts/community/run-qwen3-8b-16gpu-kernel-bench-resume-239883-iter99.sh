#!/bin/bash
# Resume task 239883 from checkpoint iter_0000099 using the original binary-reward setup.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

export RESUME_FROM="${RESUME_FROM:-${BASE_DIR}/checkpoints/qwen3_8b_kernel_bench_16gpu_20260408_095936}"
export RESUME_EXPECTED_ITER="${RESUME_EXPECTED_ITER:-99}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-qwen3_8b_kernel_bench_resume239883_iter0099}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/qwen3_8b_kernel_bench_resume239883_iter0099}"

# Match the pre-signal-enhancement binary baseline as closely as possible.
export DISABLE_EASY_INJECTION="${DISABLE_EASY_INJECTION:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
export LR="${LR:-5e-6}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-15}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-}"
export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
export DYNAMIC_SAMPLING_RESAMPLE_SAME_PROMPT_MAX_TIMES="${DYNAMIC_SAMPLING_RESAMPLE_SAME_PROMPT_MAX_TIMES:-}"
export NO_GRAD_STOP_THRESHOLD="${NO_GRAD_STOP_THRESHOLD:-}"
export NO_GRAD_STOP_PATIENCE="${NO_GRAD_STOP_PATIENCE:-}"

exec bash "${SCRIPT_DIR}/run-qwen3-8b-16gpu-kernel-bench.sh" "$@"
