#!/bin/bash
set -x
unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY
source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# Moonlight 模型路径配置
export MODEL_PATH=/user/luozhenghang/1029_miles_miniCPM8b/models/Moonlight-16B-A3B
export MEGATRON_MODEL_PATH=/user/luozhenghang/1029_miles_miniCPM8b/checkpoints/Moonlight-16B-A3B-megatron
export PYTHONPATH=.:Megatron-LM/:../mbridge

# 数据集配置 (使用 minicpm 的数据集)
TRAIN_FILE="/user/luozhenghang/1023_miles_minicpm8b/1023data/datasets/train/dapo-math-17k.jsonl"
TEST_FILE="aime2024 /user/luozhenghang/1023_miles_minicpm8b/1023data/datasets/test/aime-2024.jsonl"

export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=/data/tensorboard/moonlight_test

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/moonlight.sh"

# 检查点配置
CKPT_ARGS=(
   --hf-checkpoint $MODEL_PATH
   --load $MEGATRON_MODEL_PATH
   --save $CKPT_PATH/moonlight/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data $TRAIN_FILE
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 500
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 1024
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --global-batch-size 512
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data $TEST_FILE
   --n-samples-per-eval-prompt 8
   --eval-max-prompt-len 1024
   --eval-max-response-len 4096
   --eval-top-p 0.95
)

# 8 卡 H100 80GB 并行配置: EP=8, TP=1
PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 1
   --context-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
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
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-tensorboard
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-expert-parallel-size 8
)

mkdir -p ${CKPT_PATH}/moonlight/dump_details
MISC_ARGS=(
   --dump-details ${CKPT_PATH}/moonlight/dump_details
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # use deepep for megatron MoE
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"MASTER_PORT\": \"${MASTER_PORT}\"
  }
}"

# launch the master node of ray in container
bash scripts/setup/ray_start.sh

ray job submit --address="http://${MASTER_ADDR}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${WORLD_SIZE} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
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

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "all done!"
