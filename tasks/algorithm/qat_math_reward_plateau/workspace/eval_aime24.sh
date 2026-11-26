#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { printf 'Usage: %s CHECKPOINT OUTPUT_DIRECTORY\n' "$0" >&2; exit 2; }
workspace=$(cd "$(dirname "$0")" && pwd)
prepared=${AUTORLBENCH_PREPARED:?run the harness prepare stage first}
checkpoint=$(realpath -e "$1")
output=$(realpath -m "$2")
model="$workspace/assets/nemotron-brorl"
dataset="$workspace/data/aime24-j9073.parquet"
devices=${CUDA_VISIBLE_DEVICES:-0}
merged="$output/merged"
[[ ${checkpoint##*/} == global_step_30 ]]
test -s "$checkpoint/data.pt"
for rank in 0 1 2 3; do
    test -s "$checkpoint/actor/model_world_size_4_rank_$rank.pt"
done

cd "$workspace"
sha256sum --strict -c data/input.sha256
mkdir "$output"
export PYTHONPATH="$prepared/k2v3:$workspace/qat:$workspace/qat/vendor/verl:$prepared/accelerate-overlay:$prepared/dependencies"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 EVAL_ENFORCE_EAGER=1
cd "$workspace/qat"
CUDA_VISIBLE_DEVICES='' python3 -B -m verl.model_merger merge \
    --backend fsdp --local_dir "$checkpoint/actor" --target_dir "$merged" \
    --trust-remote-code --use_cpu_initialization > "$output/merge.log" 2>&1
for asset in "$checkpoint/actor/huggingface/"*; do
    if [[ ! -e "$merged/${asset##*/}" ]]; then
        cp "$asset" "$merged/"
    fi
done
for asset in tokenizer.json chat_template.jinja; do
    cmp "$merged/$asset" "$model/$asset"
done
for asset in config.json tokenizer_config.json special_tokens_map.json generation_config.json; do
    if [[ -f "$model/$asset" ]]; then
        cp "$model/$asset" "$merged/"
    fi
done
CUDA_VISIBLE_DEVICES="${devices%%,*}" python3 -B scripts/eval_vllm_turboquant.py \
    --benchmark aime_math --data "$dataset" --model "$merged" --output "$output/E30.json" \
    --kv-cache-dtype turboquant_k2v3_nc --n 8 --tensor-parallel-size 1 \
    --max-num-seqs 240 --max-num-batched-tokens 32768 --max-model-len 33792 \
    --gpu-memory-utilization 0.90 --seed 42 --progress > "$output/E30.log" 2>&1
