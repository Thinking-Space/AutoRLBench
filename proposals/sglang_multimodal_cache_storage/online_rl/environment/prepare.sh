#!/usr/bin/env bash
# Run inside the declared shared image. An optional wheelhouse enables offline setup.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 DATA [WHEELHOUSE]" >&2
    exit 2
fi
mkdir -p "$1"
data=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
python3 -c 'import sys, platform, torch, torchvision; assert sys.version_info[:2] == (3, 10) and platform.machine() == "x86_64"; print("Base PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)'
opts=()
if [[ $# -eq 2 ]]; then opts=(--no-index --find-links "$2"); fi
python3 -m pip --disable-pip-version-check install --no-deps --no-cache-dir --only-binary=:all: \
    --require-hashes --upgrade --target "$data/runtime" "${opts[@]}" -r "$here/requirements-linux-py310.lock"
export PYTHONPATH="$data/runtime${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import transformers; from transformers import AutoProcessor, Idefics3ForConditionalGeneration; print("Transformers:", transformers.__version__)'
python3 "$here/../prepare.py" "$data"
