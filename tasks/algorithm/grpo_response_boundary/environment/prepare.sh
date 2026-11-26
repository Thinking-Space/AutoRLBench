#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$AUTORLBENCH_PREPARED/pysite"
wheel_dir=$(mktemp -d)
trap 'rm -rf "$wheel_dir"' EXIT
python -m pip download --only-binary=:all: --no-deps --require-hashes -r "$here/python-pins.txt" -d "$wheel_dir"
while read -r name size; do
    [[ "$(wc -c < "$wheel_dir/$name")" -eq "$size" ]]
done < "$here/wheel-sizes.txt"
python -m pip install --no-index --no-deps --require-hashes -r "$here/python-pins.txt" \
    --find-links "$wheel_dir" --target "$AUTORLBENCH_PREPARED/pysite"
printf 'PYTHONPATH=%s/pysite\n' "$AUTORLBENCH_PREPARED" > "$AUTORLBENCH_PREPARED/env"

python "$here/fetch_hf.py" --type model --repo nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --dest "$AUTORLBENCH_WORKSPACE/assets/model"
python "$here/fetch_hf.py" --type dataset --repo HuggingFaceH4/aime_2024 \
  --revision e6cf0cd64082ada1c025717826bd40e155b1ec81 --dest "$AUTORLBENCH_WORKSPACE/assets/aime2024"
python "$here/split_data.py" "$AUTORLBENCH_WORKSPACE"
