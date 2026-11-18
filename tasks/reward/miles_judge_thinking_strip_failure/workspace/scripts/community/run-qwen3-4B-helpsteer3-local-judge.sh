#!/bin/bash

# GPU layout:
#   GPU 0-3: sglang judge server (Qwen3-4B, TP=1, DP=4)
#   GPU 4-7: MILES training (actor-num-gpus=4, colocate)

# --- Cleanup from previous runs ---
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

LOCAL_IP=$(hostname -I | awk '{print $1}')
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${LOCAL_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${LOCAL_IP}"
RUN_DATETIME=$(date +%Y%m%d_%H%M%S)
export TENSORBOARD_DIR=/data/tensorboard/miles_qwen3-4B-helpsteer3-local-judge/job_${JOB_ID}-${RUN_DATETIME}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# ============================================================
# 1. Launch local judge server on GPU 0-3
# ============================================================
JUDGE_MODEL=/user/lixin/models/Qwen3-4B
# JUDGE_MODEL=/user/lixin/models/Qwen2.5-72B-Instruct 
JUDGE_PORT=13142
JUDGE_LOG="/tmp/sglang_judge_${RUN_DATETIME}.log"

echo "Starting judge server (${JUDGE_MODEL}) on GPU 0-3 ..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path "${JUDGE_MODEL}" \
    --host 0.0.0.0 \
    --port "${JUDGE_PORT}" \
    --tp 2 \
    --dp-size 2 \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --skip-server-warmup \
    > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!

MAX_WAIT=300
WAITED=0
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health_generate" > /dev/null; do
    echo "Waiting for judge server to start ... (${WAITED}s elapsed)"
    tail -n 5 "${JUDGE_LOG}" 2>/dev/null || true
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Judge server failed to start within ${MAX_WAIT}s"
        tail -n 30 "${JUDGE_LOG}" 2>/dev/null || true
        exit 1
    fi
done
echo "Judge server ready at http://127.0.0.1:${JUDGE_PORT} (pid=${JUDGE_PID})"

# ============================================================
# 2. Configure training on GPU 4-7
# ============================================================
export CUDA_VISIBLE_DEVICES=4,5,6,7

CKPT_ARGS=(
   --hf-checkpoint /user/lixin/models/Qwen3-4B
   --ref-load /user/lixin/ckpts/qwen3-4b_torch_dist
   --load /user/lixin/ckpts/qwen3-4b_torch_dist
   --save /data/checkpoints
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /user/lixin/datasets/miles/helpsteer3_preference_train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type llm_judge
   --rm-url "http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions"
   --judge-model "${JUDGE_MODEL}"
   --judge-api-key "${JUDGE_API_KEY:-no_api_key_needed}"
   --judge-prompt-template "${SCRIPT_DIR}/judge_templates/helpsteer3_arenahard.txt"
   --judge-kwargs 'temperature=0.6,max_tokens=8192'
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data helpsteer3_val /user/lixin/datasets/miles/helpsteer3_preference_validation.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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

WANDB_ARGS=(
   --use-tensorboard
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================================
# 3. Launch Ray and submit training job
# ============================================================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for Ray dashboard agent to be ready (job submission fails if submitted too soon)
echo "Waiting 30s for Ray job submission API ..."
sleep 30


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"./Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

# ============================================================
# 4. Cleanup
# ============================================================
echo "Training finished. Cleaning up ..."
kill ${JUDGE_PID} 2>/dev/null || true
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
