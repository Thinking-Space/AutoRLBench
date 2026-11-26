#!/bin/bash
# GRPO on MATH (level 5) with Qwen/Qwen3-1.7B (thinking mode) in a two-turn answer-then-review environment:
# after the first answer the user asks the policy to review its solution and give a final \boxed{} answer
# (scripts/feedback_rollout.py; reward 1 if the final answer is correct).
# slime (Megatron backend + SGLang rollout), actor and rollout colocated on NGPU GPUs.
# Env: MODEL (HF checkpoint dir), TORCH_DIST (Megatron copy of MODEL; converted here if missing),
#      OUT (logs + rollout dumps), NGPU, NUM_ROLLOUT.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-$WS/models/Qwen3-1.7B}
TORCH_DIST=${TORCH_DIST:-$WS/models/Qwen3-1.7B_torch_dist}
OUT=${OUT:-$WS/outputs/grpo}
NGPU=${NGPU:-2}
NUM_ROLLOUT=${NUM_ROLLOUT:-30}
RAY_PORT=${RAY_PORT:-6479}
DASH_PORT=${DASH_PORT:-8366}
SLIME=${SLIME:-/root/slime}
MEGATRON=${MEGATRON:-/root/Megatron-LM}
mkdir -p "$OUT/rollouts"
export PYTHONUNBUFFERED=1

cd "$SLIME"
source scripts/models/qwen3-1.7B.sh

if [ ! -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]; then
  PYTHONPATH=$MEGATRON MASTER_ADDR=127.0.0.1 MASTER_PORT=${CONVERT_PORT:-29871} \
    python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL" --save "$TORCH_DIST" \
    > "$OUT/convert.log" 2>&1
fi

CKPT_ARGS=(
   --hf-checkpoint "$MODEL"
   --ref-load "$TORCH_DIST"
)

ROLLOUT_ARGS=(
   --prompt-data "$WS/data/math_l5_train.jsonl"
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 12288
   --rollout-max-context-len 24576
   --rollout-temperature 1.0
   --global-batch-size 128
   --balance-data
   --custom-generate-function-path feedback_rollout.generate
   --custom-rm-path feedback_rollout.reward_func
   --save-debug-rollout-data "$OUT/rollouts/{rollout_id}.pt"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --seed 1
)

RAY_TMP=${RAY_TMPDIR:-/tmp/ray}
trap 'pkill -f -- "$RAY_TMP/session_"; sleep 5; pkill -9 -f -- "$RAY_TMP/session_" || true' EXIT
export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1"
ray start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" --num-gpus "$NGPU" --disable-usage-stats \
  --dashboard-host 127.0.0.1 --dashboard-port "$DASH_PORT" --temp-dir "$RAY_TMP" \
  --min-worker-port ${RAY_MIN_WORKER_PORT:-30000} --max-worker-port ${RAY_MAX_WORKER_PORT:-30999} \
  --ray-client-server-port ${RAY_CLIENT_PORT:-10101} --node-manager-port ${RAY_NM_PORT:-6481} \
  --object-manager-port ${RAY_OM_PORT:-6482} --dashboard-agent-listen-port ${RAY_AGENT_PORT:-52366} \
  --metrics-export-port ${RAY_METRICS_PORT:-8091}

RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"$MEGATRON/:$WS/scripts\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"NCCL_NVLS_ENABLE\": \"0\", \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"}}"

ray job submit --address="http://127.0.0.1:$DASH_PORT" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$NGPU" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "$OUT/train.log"
