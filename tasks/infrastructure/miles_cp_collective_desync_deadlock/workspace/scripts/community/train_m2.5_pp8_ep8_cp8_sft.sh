#!/bin/bash

# MiniMax-M2.5 SFT — PP=8, EP=8, CP=8
# Based on zhangtingjian's train_m2.5_pp8_ep8_cp8.sh with tuned training hyperparams.
#
# Training hyperparams (tuned):
#   LR=5e-5, min-lr=5e-6, weight-decay=0, β1=0.9, β2=0.95
#   LR schedule: WSD (warmup-stable-decay) with exponential decay
#   GBS=128, micro-batch-size=1, clip-grad=1.0
#
# Everything else (parallelism, MoE, optimizer type, model args) kept identical
# to the reference script.
#
# Configurable env vars:
#   TRAIN_ITERS          default 250 (~3.5 epochs for 9149 samples @ GBS=128)
#   LR_WARMUP_ITERS      default 15  (~5% of 250)
#   LR_DECAY_ITERS       default 200 (iter where decay phase starts)
#   LR_WSD_DECAY_ITERS   default 50  (length of final decay phase)
#   SAVE_INTERVAL         default 50
#   ENABLE_CAPACITY       default 0
#   USE_MOCK_DATA         default 1
#   LOAD_CKPT             default 0

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0"

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-${WORLD_SIZE:-"1"}}
NODE_RANK=${NODE_RANK:-${RANK:-"0"}}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --vocab-size 200064
    --make-vocab-size-divisible-by 1
    --disable-bias-linear
    --seq-length 32768
    --max-position-embeddings 196608
    --num-layers 62
    --hidden-size 3072
    --ffn-hidden-size 1536
    --num-attention-heads 48
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 5000000
    --rotary-percent 0.5
    --norm-epsilon 1e-6
    --qk-layernorm
)

MOE_ARGS=(
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 1536
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-token-drop-policy probs
    --moe-router-dtype fp32
)

if [ "${ENABLE_CAPACITY:-0}" = "1" ]; then
    MOE_ARGS+=(
        --moe-expert-capacity-factor 1
        --moe-pad-expert-input-to-capacity
    )
fi

USE_MOCK_DATA=${USE_MOCK_DATA:-0}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/user/zhangtingjian/sonic-TE/mini_max"}
DATA_PATH=${DATA_PATH:-"/user/luozhenghang/0226_miles/data/sft/autocuda_sft_glm5.jsonl"}

if [ "${USE_MOCK_DATA}" = "1" ]; then
    DATA_ARGS=(
        --tokenizer-type NullTokenizer
        --tokenizer-model ${TOKENIZER_PATH}
        --split 99990,8,2
        --mock-data
    )
else
    DATA_ARGS=(
        --sft
        --tokenizer-type NullTokenizer
        --tokenizer-model ${TOKENIZER_PATH}
        --data-path ${DATA_PATH}
        --split 99990,8,2
    )
fi

LOAD_CKPT=${LOAD_CKPT:-1}
MEGATRON_CKPT_DIR=${MEGATRON_CKPT_DIR:-"/user/luozhenghang/0226_miles/checkpoints/minimax-m2.5-full-megatron"}
CKPT_ARGS=()

if [ "${LOAD_CKPT}" = "1" ]; then
    if [ ! -f "${MEGATRON_CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "============================================================"
        echo "ERROR: Megatron checkpoint not found at ${MEGATRON_CKPT_DIR}"
        echo "Please convert the HF checkpoint first."
        echo "============================================================"
        exit 1
    fi
    CKPT_ARGS+=(--load ${MEGATRON_CKPT_DIR})
    echo "Loading Megatron checkpoint from ${MEGATRON_CKPT_DIR}"
else
    echo "Using random weight initialization (LOAD_CKPT=0)"
fi

SAVE_CKPT_DIR=${SAVE_CKPT_DIR:-"/user/luozhenghang/0226_miles/checkpoints/minimax_m2.5_sft_megatron"}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
CKPT_ARGS+=(--save ${SAVE_CKPT_DIR} --save-interval ${SAVE_INTERVAL})
mkdir -p ${SAVE_CKPT_DIR}
echo "Saving checkpoints to ${SAVE_CKPT_DIR} every ${SAVE_INTERVAL} iters"

TRAIN_ITERS=${TRAIN_ITERS:-250}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-15}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-200}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-50}

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 5e-5
    --min-lr 5e-6
    --train-iters ${TRAIN_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --weight-decay 0.0
    --clip-grad 1.0
    --bf16
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-size 8
    --decoder-first-pipeline-num-layers 7
    --decoder-last-pipeline-num-layers 7
    --expert-model-parallel-size 8
    --context-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 1000
    --eval-iters 10
    --log-throughput
    --log-task-loss-interval -1
    --async-save
    --ckpt-format torch_dist
)

echo "============================================================"
echo "MiniMax-M2.5 SFT | PP=8, EP=8, CP=8 | 256 experts / top-8"
echo "LR=5e-5, min-lr=5e-6, WSD schedule, weight-decay=0, β2=0.95"
echo "GBS=128, train-iters=${TRAIN_ITERS}, warmup=${LR_WARMUP_ITERS}, decay-start=${LR_DECAY_ITERS}, decay-len=${LR_WSD_DECAY_ITERS}"
echo "mock=${USE_MOCK_DATA} | load_ckpt=${LOAD_CKPT} | moe_capa=${ENABLE_CAPACITY:-0}"
echo "GPUs: ${NNODES} nodes × ${GPUS_PER_NODE} GPUs | Rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================================"

set -ex
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_minicpm.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    $@
