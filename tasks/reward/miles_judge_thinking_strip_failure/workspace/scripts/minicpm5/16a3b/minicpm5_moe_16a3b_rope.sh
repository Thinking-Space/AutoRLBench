# MiniCPM5 MoE 16-A3B — aligned with job_362482.step_47000/config.json
# Architecture: 28 layers, 160 routed experts + 1 shared expert, topk=16
# Attention: GQA (32 heads, 2 KV heads), head_dim 128, Gated Attention
# Dense FFN: intermediate_size 8192; MoE expert FFN: moe_intermediate_size 512
# Position Encoding: standard RoPE (rope_theta=5000000, rope_scaling=null, max_position_embeddings=524288)

# MoE Configuration
MOE_SHARED_EXPERTS=1
MOE_FFN_HIDDEN=512
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
MOE_ROUTER_TOPK_SCALING_FACTOR=3.66
NLAYERS=28
FIRST_K_DENSE_REPLACE=1

# Build MOE_LAYER_FREQ array: 0 for dense layers, 1 for MoE layers
arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# MiniCPM5 MoE Model Arguments
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --vocab-size 130560
    --make-vocab-size-divisible-by 64

    # Gated Attention
    --attention-output-gate

    # RoPE Configuration (standard RoPE, no scaling)
    --rotary-base 5000000
    --max-position-embeddings 524288
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-rope-fusion

    # MoE Configuration
    --num-experts 160
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk 16
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor $MOE_ROUTER_TOPK_SCALING_FACTOR
    --moe-token-drop-policy probs
    --moe-router-dtype fp32
    --moe-permute-fusion
)
