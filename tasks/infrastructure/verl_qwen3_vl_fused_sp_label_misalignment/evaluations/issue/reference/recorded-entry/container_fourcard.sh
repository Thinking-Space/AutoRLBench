#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/test/test1708/gy/experiment/issue6068-quality-bench
IMAGE=/home/test/test1708/gy/experiment/issue6609/images/verl-rootfs
cd "$ROOT"
test "$(pwd -P)" = "$ROOT"
mode=${1:?cpu or gpu}; shift
variant=${BENCH_SOURCE_VARIANT:-before}
case "$variant" in before|after) ;; *) exit 2 ;; esac
devices=''
case "$mode" in
 cpu) ;;
 gpu) devices=${BENCH_GPUS:?allocation required}; test -s "$ROOT/metadata/gpu-allocation.json"; test -s "$ROOT/metadata/ready.json" ;;
 *) exit 2 ;;
esac
for name in home tmp vartmp shm runtime cache/singularity; do mkdir -p "$ROOT/$name"; done
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export SINGULARITY_TMPDIR="$ROOT/tmp" SINGULARITY_CACHEDIR="$ROOT/cache/singularity"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
exec singularity exec --nv --cleanenv --contain --no-mount hostfs,bind-paths \
 --home "$ROOT/home:$ROOT/home" --workdir "$ROOT/runtime" \
 --bind "$ROOT:$ROOT:rw" --bind "$ROOT/tmp:/tmp:rw" \
 --bind "$ROOT/vartmp:/var/tmp:rw" --bind "$ROOT/shm:/dev/shm:rw" \
 --pwd "$ROOT" "$IMAGE" \
 env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
 HOME="$ROOT/home" TMPDIR=/tmp XDG_CACHE_HOME="$ROOT/cache" \
 PATH="$ROOT/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
 PYTHONPATH="$ROOT/src/$variant:$ROOT/scripts" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
 HF_HOME="$ROOT/cache/huggingface" HF_DATASETS_CACHE="$ROOT/cache/datasets" HF_HUB_DISABLE_XET=1 \
 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TORCH_HOME="$ROOT/cache/torch" \
 TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" TRITON_CACHE_DIR="$ROOT/cache/triton" \
 CUDA_CACHE_PATH="$ROOT/cache/cuda" VLLM_CACHE_ROOT="$ROOT/cache/vllm" \
 PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR="$ROOT/cache/pip" \
 UV_CACHE_DIR="$ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/cache/uv-python" \
 RAY_TMPDIR=/tmp/ray CUDA_VISIBLE_DEVICES="$devices" TOKENIZERS_PARALLELISM=false \
 BENCH_FOURCARD_DIR="${BENCH_FOURCARD_DIR:-}" BENCH_STAGE_STOP="${BENCH_STAGE_STOP:-0}" BENCH_AUDIT_DIR="${BENCH_AUDIT_DIR:-}" \
 BENCH_GPU_DEADLINE="${BENCH_GPU_DEADLINE:-0}" \
 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
 RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1 "$@"
