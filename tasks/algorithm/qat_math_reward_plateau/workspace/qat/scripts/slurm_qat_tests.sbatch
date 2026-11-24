#!/bin/bash
#SBATCH --job-name=qat-parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/historical/shared/qat/r/qat-tests/parity-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-tests/parity-%j.err
# Parity gate for the TurboQuant fake quantizer.  Runs the deployed Triton
# store kernel and compares its integer quantization decisions against
# qat_kv.turboquant_fake_quant.  Must pass before any QAT run is meaningful.
set -euo pipefail

# The ccremote agent shell exports LD_PRELOAD=libproxychains4.so; with
# --export=ALL that shim is injected into every job process and hangs Ray
# actor startup (TaskRunner never connects).  Strip it unconditionally.
unset LD_PRELOAD
ROOT=/historical/home/QAT
PY=/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
# Triton JIT-compiles a small cuda_utils.c at first launch and needs
# Python.h, which is not in the base image; the main MOPD launcher uses
# the same vendored python-devel tree.
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PYINC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"
cd "$ROOT"
# verl -> ray -> msgpack, and msgpack ships only in the train overlay.
export PYTHONPATH=$ROOT/tests/_pytest_shim:$ROOT:$ROOT/vendor/verl:/historical/shared/qat/tmp/turboquant-train-overlay:/historical/shared/qat/envs/mopd-pylibs
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
$PY tests/_run_no_pytest.py \
  tests/test_turboquant_fake_quant.py \
  tests/test_turboquant_attn_hook.py \
  tests/test_tq_scheme.py \
  tests/test_block_align.py \
  tests/test_block_align_qat.py \
  tests/test_turboquant_parity.py \
  tests/test_teacher_replay.py
