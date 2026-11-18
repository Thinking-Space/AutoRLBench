#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Ensure local connections bypass HTTP proxy
LOCAL_IP=$(hostname -I | awk '{print $1}')
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${LOCAL_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${LOCAL_IP}"
RUN_DATETIME=$(date +%Y%m%d_%H%M%S)
export TENSORBOARD_DIR=/data/tensorboard/miles_qwen3-4B-helpsteer3/job_${JOB_ID}-${RUN_DATETIME}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

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
   --rm-url https://yeysai.com/v1/chat/completions
   --judge-model kimi-k2-instruct-0905
   --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY env var}"
   --judge-proxy http://whitelist-proxy.cybertron.svc.cluster.local:7891
   --judge-prompt-template "${SCRIPT_DIR}/judge_templates/helpsteer3_arenahard.txt"
   --judge-kwargs 'temperature=0.6,max_tokens=8192'
   --num-rollout 200
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 128
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

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
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
   --actor-num-gpus-per-node 8 \
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