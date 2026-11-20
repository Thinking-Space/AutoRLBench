#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model="$(realpath "${1:?pass the local model directory}")"
out="$(realpath "${2:?pass the run output directory}")"
steps="${3:-16}"
mkdir -p "$out"
export ROUTING_OUTPUT="$out"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export RAY_USAGE_STATS_ENABLED=0
export no_proxy="127.0.0.1,localhost,$(hostname -i | awk '{print $1}')"
export NO_PROXY="$no_proxy"
export HF_HOME="$out/hf-cache" TRITON_CACHE_DIR="$out/triton" TORCHINDUCTOR_CACHE_DIR="$out/torchinductor"

megatron="${MEGATRON_DIR:-}"
if [[ -z "$megatron" ]]; then
    for candidate in /root/Megatron-LM /opt/Megatron-LM /sgl-workspace/Megatron-LM; do
        if [[ -d "$candidate/megatron" ]]; then megatron="$candidate"; break; fi
    done
fi
test -n "$megatron"
export PYTHONPATH="$workspace/slime:$workspace:$megatron"
cd "$workspace/slime"
source scripts/models/qwen3.5-0.8B.sh

python3 - <<'PY'
import importlib.metadata as m
import slime
import torch
assert torch.cuda.device_count() == 1
for package in ("torch", "sglang", "ray", "megatron-core", "transformers", "flash-linear-attention"):
    try:
        print(f"runtime {package} {m.version(package)}")
    except m.PackageNotFoundError:
        print(f"runtime {package} unavailable")
print("Slime package", slime.__file__)
print("GPU", torch.cuda.get_device_name())
PY

python3 -m routing.dataset "$out/data"
reference="$out/reference"
torchrun --standalone --nproc_per_node=1 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --bf16 --hf-checkpoint "$model" --save "$reference" \
    --attention-backend flash --attention-dropout 0 --hidden-dropout 0 \
    2>&1 | tee "$out/convert.log"

ray start --head --node-ip-address 127.0.0.1 --port 6479 --dashboard-port 8277 \
    --dashboard-host 127.0.0.1 --num-gpus 1 --num-cpus 8 --disable-usage-stats
trap 'ray stop >/dev/null 2>&1 || true' EXIT
python3 - "$out/ray-env.json" <<'PY'
import json
import os
import sys
from pathlib import Path
keys = ("PYTHONPATH", "ROUTING_OUTPUT", "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE", "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "no_proxy", "NO_PROXY")
Path(sys.argv[1]).write_text(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))
PY

ray job submit --address http://127.0.0.1:8277 --runtime-env "$out/ray-env.json" -- \
    python3 train.py "${MODEL_ARGS[@]}" \
    --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 --colocate \
    --bf16 --hf-checkpoint "$model" --ref-load "$reference" \
    --prompt-data "$out/data/train.jsonl" --input-key prompt --label-key label --metadata-key metadata \
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' --rollout-shuffle \
    --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 8 \
    --num-rollout "$steps" --rollout-temperature 0.8 --rollout-top-p 0.95 \
    --rollout-max-response-len 384 --rollout-max-context-len 1024 \
    --rollout-stop-token-ids 248044 248046 \
    --eval-prompt-data validation "$out/data/validation.jsonl" --eval-interval 4 \
    --n-samples-per-eval-prompt 1 --eval-temperature 0.0 \
    --custom-generate-function-path routing.rollout.generate \
    --custom-reward-post-process-path routing.reward.post_process_rewards \
    --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed 1197 --rollout-seed 1197 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 1024 --seq-length 1024 \
    --attention-backend flash --attention-dropout 0 --hidden-dropout 0 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
    --sglang-context-length 1024 --sglang-max-running-requests 8 --sglang-cuda-graph-max-bs 8 \
    --save "$out/checkpoints" --save-interval 4 --no-save-optim --no-save-rng \
    --save-debug-rollout-data "$out/dumps/rollout_{rollout_id}.pt"
