#!/usr/bin/env bash
# Qwen2.5 GSM8K prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"
W="$AUTORLBENCH_WORKSPACE"
E="$AUTORLBENCH_ENVIRONMENT"
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$AUTORLBENCH_PREPARED/pip-cache"
export HF_HOME="$AUTORLBENCH_PREPARED/huggingface"
export HF_HUB_DISABLE_XET=1
python3 "$E/prepare_runtime.py"
P="$AUTORLBENCH_PREPARED/runtime/bin/python"

fetch_repo() {
    if "$P" "$E/fetch_assets.py" "$@"; then
        return
    fi
    HF_ENDPOINT=https://hf-mirror.com "$P" "$E/fetch_assets.py" "$@"
}

# Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
fetch_repo --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --dest "$W/model/Qwen2.5-1.5B-Instruct"
# LIF1014/qwen25-gsm8k-grpo @ 1d60021828cd3d735056165ffcb587408caa16ac
fetch_repo --type dataset --repo LIF1014/qwen25-gsm8k-grpo \
    --revision 1d60021828cd3d735056165ffcb587408caa16ac \
    --dest "$W/data/snapshot"
"$P" "$E/check_assets.py"
"$P" "$E/materialize_inputs.py"
"$P" "$E/stage_inputs.py"
"$P" "$E/build_data.py"
