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
export MODEL_PATH=/user/panyinxu/models/Qwen3-8B
export MEGATRON_MODEL_PATH=/user/panyinxu/ckpts/qwen3-8b_torch_dist
export PYTHONPATH=.:/root/Megatron-LM/
TRAIN_FILE="/user/panyinxu/datasets/miles_data/dapo-math-17k.jsonl \
/user/panyinxu/datasets/miles_data/deepmath.jsonl \
/user/panyinxu/datasets/miles_data/deepscaler.jsonl \
/user/panyinxu/datasets/miles_data/codeforces_hard_09.jsonl \
/user/panyinxu/datasets/miles_data/leetcode_hard_09.jsonl \
/user/panyinxu/datasets/miles_data/prime_codecontests_09.jsonl \
/user/panyinxu/datasets/miles_data/prime_codeforces_09.jsonl"
TEST_FILE="aime2025 /user/panyinxu/datasets/miles_data/aime-2025.jsonl \
aime2024 /user/panyinxu/datasets/miles_data/aime-2024.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=/data/tensorboard/
# export SGLANG_DUMPER_ENABLE=1
# export SGLANG_DUMPER_DIR=/user/panyinxu/share_user_rl/panyinxu/miles/sglang_dump
# export SGLANG_DUMPER_WRITE_FILE=1
# mkdir -p $SGLANG_DUMPER_DIR

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

# (cd /sgl-workspace/sglang && git status)
# cat /sgl-workspace/sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py

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
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint $MODEL_PATH
   # --ref-load /root/Qwen3-32B_torch_dist/
   --load $MEGATRON_MODEL_PATH
   --save $CKPT_PATH
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data $TRAIN_FILE
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 1000
   --rollout-batch-size 128
   --n-samples-per-prompt 16
   --rollout-max-prompt-len 1000
   --rollout-max-response-len 39000
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --global-batch-size 2048
   --balance-data
   --use-miles-router
   --use-distributed-post
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data $TEST_FILE
   --n-samples-per-eval-prompt 32
   --eval-max-prompt-len 1000
   --eval-max-response-len 39000
   --eval-top-p 1.0
)

PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 8
   --context-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10000
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   --use-tensorboard
   # --wandb-project miles-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
   # --sglang-context-length 65536
   # --sglang-json-model-override-args '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}'
   # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-enable-ep-moe
)

mkdir -p ${CKPT_PATH}/$RUN_ID/dump_details
MISC_ARGS=(
   --dump-details ${CKPT_PATH}/$RUN_ID/dump_details
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --no-gradient-accumulation-fusion
   # --use-rope-scaling
   # --rope-scaling-factor 4.0
   # --max-position-embeddings 40960
)

CUSTOM_ARGS=(
   # Use DrGRPO loss reducer
   # --custom-pg-loss-reducer-function-path examples.DrGRPO.custom_reducer.get_pg_loss_reducer
   # TIS-related args, recommended to enable when using TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

TRUE_ON_POLICY_ARGS=(
   --sglang-enable-deterministic-inference
   --sglang-rl-on-policy-target fsdp
   --sglang-attention-backend fa3
   --deterministic-mode
   # # fsdp
   # --attn-implementation flash_attention_3
   # --true-on-policy-mode
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"MASTER_PORT\": \"${MASTER_PORT}\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\"
  }
}"

# launch the master node of ray in container
bash scripts/setup/ray_start.sh

if [ $RANK -eq 0 ]; then
   ray job submit --address="http://127.0.0.1:8265" \
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
      ${MISC_ARGS[@]} \
      ${CUSTOM_ARGS[@]} \
      ${TRUE_ON_POLICY_ARGS[@]}
fi

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "all done!"