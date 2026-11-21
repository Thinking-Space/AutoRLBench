#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
workspace="$PWD"
prepared="${AUTORLBENCH_PREPARED:?run environment/prepare.sh first}"
output="${LOG_DIR:-$workspace/outputs}"
mkdir -p "$output"
sha256sum --strict --check recorded.sha256 data/input.sha256
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
run_segment() {
    local endpoint="$1" destination="$2"
    shift 2
    python3 -B qat_historical_reward_two_gpu_launch.py 9073 --source "$workspace/qat" \
        --assets "$workspace/assets" --recorded "$workspace/j9073-effective.json" \
        --recorded-sha256 "$(cut -d ' ' -f1 recorded.sha256)" \
        --reference "$workspace/j9073-compose-legacy-data.stdout" --output "$destination" \
        --dependencies "$prepared/accelerate-overlay:$prepared/dependencies" \
        --k2v3-overlay "$prepared/k2v3" --reward-qualification-steps "$endpoint" --reward-two-gpu "$@" --run
}
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--short" ]; }; then
    printf 'Usage: %s [--short]\n' "$0" >&2; exit 2
fi
run_segment 8 "$output/steps-01-08"
if [ "${1:-}" = "--short" ]; then exit 0; fi
run_segment 16 "$output/steps-09-16" --reward-resume "$output/steps-01-08/checkpoints/global_step_8"
run_segment 24 "$output/steps-17-24" --reward-resume "$output/steps-09-16/checkpoints/global_step_16"
