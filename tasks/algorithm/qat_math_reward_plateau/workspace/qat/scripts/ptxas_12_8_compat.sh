#!/usr/bin/env bash
# Triton 3.x in the pinned vLLM environment rejects CUDA 13's version string,
# while the L20D node only exposes CUDA 13 ptxas.  Report the torch-compatible
# version to Triton but execute the real CUDA 13 assembler for all commands.
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "Cuda compilation tools, release 12.8, V12.8.89"
  exit 0
fi
exec /usr/local/cuda/bin/ptxas "$@"
