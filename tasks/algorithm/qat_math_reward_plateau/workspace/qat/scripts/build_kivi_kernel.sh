#!/usr/bin/env bash
set -euo pipefail

# Build the CUDA extension from the pinned official KIVI checkout.  No local
# reimplementation is used: qat_kv.kivi_qwen2 imports quant/new_pack.py and
# quant/matmul.py from this checkout at runtime.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIVI_ROOT="${KIVI_ROOT:-/historical/shared/qat/third_party/KIVI}"
KIVI_COMMIT="876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
PYTHON_BIN="${PYTHON_BIN:-/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python}"

if [[ ! -d "${KIVI_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${KIVI_ROOT}")"
  git clone https://github.com/jy-yuan/KIVI.git "${KIVI_ROOT}"
fi
git -C "${KIVI_ROOT}" fetch --quiet origin "${KIVI_COMMIT}"
git -C "${KIVI_ROOT}" checkout --quiet --detach "${KIVI_COMMIT}"

if [[ ! -f "${KIVI_ROOT}/quant/setup.py" ]]; then
  echo "official KIVI checkout is missing quant/setup.py: ${KIVI_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${KIVI_ROOT}:${KIVI_ROOT}/quant:${PYTHONPATH:-}"
cd "${KIVI_ROOT}/quant"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import setuptools, torch' >/dev/null 2>&1; then
  echo "PYTHON_BIN must provide setuptools and torch: ${PYTHON_BIN}" >&2
  exit 2
fi

# The cluster CUDA toolkit can be newer than the CUDA version used to build
# torch.  The official KIVI sources remain unchanged; this only bypasses
# PyTorch's conservative version check after the caller explicitly opts in.
if [[ "${ALLOW_CUDA_VERSION_MISMATCH:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -c 'import runpy,sys,torch.utils.cpp_extension as c; c._check_cuda_version=lambda *a,**k: None; runpy.run_path("setup.py", run_name="__main__")' build_ext --inplace
else
  "${PYTHON_BIN}" setup.py build_ext --inplace
fi
echo "built official KIVI ${KIVI_COMMIT} at ${KIVI_ROOT}"
