#!/usr/bin/env bash
# Submit eval scheduler — defective version.
# BUG: infer_type=api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1 is not registered in UltraEval2.
# Every submission returns: submit_failed: 无可用资源池

set -ex

# BUG: this infer_type is NOT registered in UltraEval2's inference pool configuration.
# Fix: use api_general_think_sft_t1_tp0.95_rp1.05
INFER_TYPE=${INFER_TYPE:-api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1}

# Submit three benchmark eval job
python3 -m evalhub.platform.scheduler \
  --state_file outputs/scheduler_state.json \
  --mode api-eval \
  --infer_type "${INFER_TYPE}" \
  --model_path /models/hf-1594562-iter149 \
  --datasets "arena_hard_v2_...,writingbench_...,tigan_..." \
  "$@"
