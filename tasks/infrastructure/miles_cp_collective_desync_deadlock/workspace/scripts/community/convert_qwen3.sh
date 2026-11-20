set -x
source scripts/models/qwen3-4B.sh
# export PYTHONPATH=./Megatron-LM/:./mbridge:.
export PYTHONPATH=./Megatron-LM:.
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /user/lixin/models/Qwen3-4B \
    --save /user/lixin/ckpts/qwen3-4b_torch_dist/ \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 4 \
    ${MODEL_ARGS[@]}
