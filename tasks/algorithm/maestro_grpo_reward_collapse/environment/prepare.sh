#!/usr/bin/env bash
# Download the dataset, frozen tool results and all seven model asset sets.
# Only this stage may contact Hugging Face. No host conda or host model paths.
set -euo pipefail
environment_dir="${AUTORLBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must name the staged workspace}"
export PYTHONDONTWRITEBYTECODE=1
python3 "$environment_dir/restore_logs.py" --workspace "$workspace"
python3 "$environment_dir/prepare_data.py" --workspace "$workspace"
python3 "$environment_dir/prepare_models.py" --workspace "$workspace"
printf '%s\n' '[prepare] Historical logs restored; dataset, tool inputs and all seven models verified.'
