set -x
source scripts/models/qwen3-30B-A3B.sh
# export PYTHONPATH=./Megatron-LM/:./mbridge:.
export PYTHONPATH=/root/Megatron-LM:.
export WORLD_SIZE=1
export RANK=0

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /user/panyinxu/ckpts/qwen3-30a3b-sft-step1500 \
    --save /user/panyinxu/ckpts/qwen3-30a3b_sft_step1500_torch_dist/ \
    ${MODEL_ARGS[@]}
