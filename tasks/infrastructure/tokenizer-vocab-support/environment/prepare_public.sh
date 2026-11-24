#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${AUTORLBENCH_WORKSPACE:?prepare requires the harness workspace path}"
: "${AUTORLBENCH_PREPARED:?prepare requires the harness prepared path}"
: "${AUTORLBENCH_MODELS:?prepare requires the model download cache}"
: "${AUTORLBENCH_DATASETS:?prepare requires the dataset download cache}"
python "$HERE/prepare_fetch.py" --manifest "$HERE/model.assets.json" \
    --dest "$AUTORLBENCH_WORKSPACE/models/actor"
python "$HERE/prepare_fetch.py" --manifest "$HERE/train-raw.assets.json" \
    --dest "$AUTORLBENCH_PREPARED/skywork-raw"
python "$HERE/prepare_data.py" --workspace "$AUTORLBENCH_WORKSPACE" \
    --raw "$AUTORLBENCH_PREPARED/skywork-raw" --assets "$HERE/assets"
mkdir -p "$AUTORLBENCH_PREPARED/wheels" "$AUTORLBENCH_PREPARED/pysite"
# codex: pin the two pure-Python grader dependencies used in the shared-image run.
python -m pip download --no-deps --dest "$AUTORLBENCH_PREPARED/wheels" \
    math-verify==0.9.0 latex2sympy2-extended==1.11.0
printf '%s  %s\n' \
    3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194 "$AUTORLBENCH_PREPARED/wheels/math_verify-0.9.0-py3-none-any.whl" \
    aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234 "$AUTORLBENCH_PREPARED/wheels/latex2sympy2_extended-1.11.0-py3-none-any.whl" | sha256sum -c -
python "$HERE/stage_accelerate_compat.py" --target "$AUTORLBENCH_PREPARED/pysite"
python -m pip install --no-index --no-deps --target "$AUTORLBENCH_PREPARED/pysite" \
    "$AUTORLBENCH_PREPARED/wheels/math_verify-0.9.0-py3-none-any.whl" \
    "$AUTORLBENCH_PREPARED/wheels/latex2sympy2_extended-1.11.0-py3-none-any.whl"
printf '%s\n' "PYTHONPATH=$AUTORLBENCH_PREPARED/pysite:$AUTORLBENCH_WORKSPACE/framework" \
    'WANDB_MODE=offline' > "$AUTORLBENCH_PREPARED/env"
