# MiniCPM5-1B model args (Llama dense architecture)
# Source:
# /user/panyinxu/models/MiniCPM5-1B-SFT/config.json
MODEL_ARGS=(
   --swiglu
   --disable-bias-linear
   --num-layers 24
   --hidden-size 1536
   --ffn-hidden-size 4608
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128
   --vocab-size 130560
   --make-vocab-size-divisible-by 64
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-5000000}"
   --max-position-embeddings 131072
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --untie-embeddings-and-output-weights
   --no-masked-softmax-fusion
   --no-rope-fusion
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)
