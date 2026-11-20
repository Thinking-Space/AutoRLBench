unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY
unset PYTORCH_CUDA_ALLOC_CONF

export WORK_DIR=`pwd`
export PYTHONUNBUFFERED=1
export HF_TRUST_REMOTE_CODE="1"
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_NSOCKS_PERTHREAD=8
# export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=/root/.cache/huggingface/modules:${WORK_DIR}/Megatron-LM:${PYTHONPATH:-}
export USE_FP8_GEMM=0
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
if [ -z "${GPUS_PER_NODE+x}" ]; then
    export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    export GPUS_PER_NODE="$GPUS_PER_NODE"
fi

# If RANK does not exist, set default values for distributed training
if [ -z "${RANK+x}" ]; then
    export RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=$(hostname)
    export MASTER_PORT=23456
fi

if [ -z "${ACTOR_NUM_NODES+x}" ]; then
    export ACTOR_NUM_NODES=$((WORLD_SIZE / 2))
else
    export ACTOR_NUM_NODES="$ACTOR_NUM_NODES"
fi
# ROLLOUT_NUM_GPUS = (WORLD_SIZE - ACTOR_NUM_NODES) * GPUS_PER_NODE
ROLLOUT_NUM_GPUS=$(((WORLD_SIZE - ACTOR_NUM_NODES) * GPUS_PER_NODE))

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN # TRACE or INFO or WARN
# export VLLM_TRACE_FUNCTION=1
# check IB status
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_CGRAPH_submit_timeout=3000
export RAY_CGRAPH_get_timeout=3000
echo "PyTorch NCCL version:"
python -c "import torch; print(torch.cuda.nccl.version())"
echo "NCCL related environment variables:"
env | grep NCCL
env | grep -E "NCCL*|LD_LIBRARY_PATH"
export PROJECT_NAME='miles_grpo_minicpm4_rl_scaling'
export EXPERIMENT_NAME='minicpm4_8b_scaling'
export SFT_MODEL_PATH=${SFT_MODEL_PATH:-/user/panyinxu/models/MiniCPM4.1-8B}
export CKPT_PATH=${CKPT_PATH:-/data/checkpoints/${PROJECT_NAME}-${PROJECT_ID}/${JOB_ID}/checkpoints/}
export ROLLOUT_DATA_DIR=$CKPT_PATH/rollout_data
mkdir -p $CKPT_PATH
mkdir -p $ROLLOUT_DATA_DIR
export TENSORBOARD_DIR=/data/tensorboard/$PROJECT_NAME/$EXPERIMENT_NAME
mkdir -p $TENSORBOARD_DIR
export MLFLOW_TRACKING_URI=sqlite:////data/checkpoints/mlruns.db
echo $PATH
echo $LD_LIBRARY_PATH
echo $NCCL_IB_HCA

ulimit -n 1048576 || true

hostname -i
nvidia-smi
free -h

export SWANLAB_API_KEY=86XgDZ8lOTIZqhEzPa26X
export SWANLAB_LOG_DIR=/data/tensorboard/swanlab
mkdir -p $SWANLAB_LOG_DIR
export SWANLAB_MODE=cloud

source $WORK_DIR/scripts/setup/proxy.sh

mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${SFT_MODEL_PATH}/*.py /root/.cache/huggingface/modules/transformers_modules/

export RAY_TMPDIR=/data/logs/tmp/ray/rank_${RANK}
# export RAY_LOG_TO_STDERR=1
mkdir -p $RAY_TMPDIR
export TMPDIR=/data/logs/tmp
mkdir -p $TMPDIR

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache/${JOB_ID:-default}_rank${RANK}}
mkdir -p $TRITON_CACHE_DIR
export PIP_CACHE_DIR=/data/logs/tmp/pip_cache
mkdir -p $PIP_CACHE_DIR
export HF_HOME=/data/logs/tmp/huggingface
mkdir -p $HF_HOME
export TORCH_EXTENSIONS_DIR=/data/logs/tmp/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR
export TORCH_HOME=/data/logs/tmp/torch_home
mkdir -p $TORCH_HOME

export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=5000
