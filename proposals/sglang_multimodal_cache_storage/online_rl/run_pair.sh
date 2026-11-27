#!/usr/bin/env bash
# DATA contains the weights, upstream sources, and the optional Linux package overlay.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 DATA OUTPUT [cuda|mps|cpu]" >&2
    exit 2
fi
data=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
device=${3:-cuda}
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$data/runtime${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
for arm in buggy fixed; do
    python3 -u "$here/train_vlm.py" --device "$device" --mode "$arm" --model-dir "$data/smolvlm" \
        --source-dir "$data/upstream" --output "$out/$arm.jsonl" 2>&1 | tee "$out/$arm.log"
done
python3 "$here/verify_pair.py" "$out/buggy.jsonl" "$out/fixed.jsonl" --out "$out/paired.json"
