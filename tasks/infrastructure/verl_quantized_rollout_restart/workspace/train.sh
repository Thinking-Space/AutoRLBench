#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
workspace="$PWD"
prepared="${AUTORLBENCH_PREPARED:?run environment/prepare.sh first}"
logs="${LOG_DIR:-$workspace/outputs}"
mkdir -p "$logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$workspace/qat/vendor/verl:$workspace/qat:$prepared/accelerate-overlay:$prepared/dependencies"
sha256sum --strict --check data/input.sha256

nvfp4="$logs/nvfp4"
if [ ! -f "$nvfp4/.complete" ]; then
    attempt="$(mktemp -d "$logs/.calibrate-XXXXXX")"
    python3 -B -u qat/scripts/ptq_nvfp4.py --model "$workspace/models/student" \
        --tier w4a4 --out "$attempt" --calib-data "$workspace/data/teacher-calibration-prefix.parquet" \
        --calib-samples 1000 --calib-seq-len 2048 --calib-batch 2
    test -s "$attempt/modelopt_state.pt"
    printf '%s  %s\n' 98fd1832af04338afc5e689d5382aa84663ec77378647b1d6a6483b80581d95b \
        "$attempt/model.safetensors" | sha256sum --strict --check -
    touch "$attempt/.complete"
    mv -T "$attempt" "$nvfp4"
fi
export OMP_NUM_THREADS=4
exec python3 -B launch.py --output "$logs" --nvfp4 "$nvfp4"
