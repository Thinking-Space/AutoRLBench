#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd "$(dirname "$0")" && pwd)
prepared=${AUTORLBENCH_PREPARED:?run the harness prepare stage first}
output=${LOG_DIR:-$workspace/outputs}
mkdir -p "$output"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4

cd "$workspace"
sha256sum --strict --check recorded.sha256 data/input.sha256 overlay.sha256
if [[ ${1:-} == --compose-only && $# == 1 ]]; then
    exec python3 -B run_math_rl.py "$workspace" "$prepared" "$output/compose" \
        --steps 8 --historical --response-limit 16384 --compose-only
fi
[[ $# == 0 ]]

first="$output/steps-01-08"
mkdir "$first"
python3 -B run_math_rl.py "$workspace" "$prepared" "$first/run" \
    --steps 8 --historical --response-limit 16384
printf '0\n' > "$first/exit_code"

last="$output/steps-09-30"
mkdir "$last"
python3 -B run_math_rl.py "$workspace" "$prepared" "$last/run" \
    --steps 30 --historical --response-limit 16384 \
    --resume "$first/run/checkpoints/global_step_8"
printf '0\n' > "$last/exit_code"
