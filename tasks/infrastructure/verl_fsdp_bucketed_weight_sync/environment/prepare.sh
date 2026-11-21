#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -f "$site/.complete" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        numpy==1.26.4 scipy==1.17.1 pyarrow==19.0.1 datasets==4.0.0
    touch "$site/.complete"
fi
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env"; then
    echo "PYTHONPATH=$site:$W" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import datasets, numpy, pyarrow, scipy
assert numpy.__version__ == "1.26.4", numpy.__version__
assert scipy.__version__ == "1.17.1", scipy.__version__
assert pyarrow.__version__ == "19.0.1", pyarrow.__version__
assert datasets.__version__ == "4.0.0", datasets.__version__
EOF

# Pinned HF revision: Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --allow LICENSE \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expect-size model.safetensors=3087467144 \
    --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
    --expect-size tokenizer.json=7031645 \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"

# Pinned HF revision: Jiayi-Pan/Countdown-Tasks-3to4 @ 408f70d177020686d34a56bba5952feb45aaaee4
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo Jiayi-Pan/Countdown-Tasks-3to4 \
    --type dataset \
    --revision 408f70d177020686d34a56bba5952feb45aaaee4 \
    --allow README.md \
    --allow data/train-00000-of-00001.parquet \
    --expect README.md=eb56eda2e272457a5c87af03492ca62b966db69117bd6e172d8f44c93b79c373 \
    --expect-size README.md=314 \
    --expect data/train-00000-of-00001.parquet=a1b37bea0f439c858b11b908daa682e521f13f9592f1866b08650f58b730d126 \
    --expect-size data/train-00000-of-00001.parquet=2845904 \
    --dest "$W/raw/countdown"
