#!/usr/bin/env bash
set -euo pipefail

environment=$(cd "$(dirname "$0")" && pwd)
: "${AUTORLBENCH_WORKSPACE:?prepare requires the harness workspace path}"
bash "$environment/prepare_public.sh"
# codex: the staged model does not use the Hugging Face cache metadata.
rm -rf -- "$AUTORLBENCH_WORKSPACE/models/actor/.cache/huggingface"
if [[ ${AUTORLBENCH_EVALUATION_LEVEL:-solution} == solution ]]; then
    bash "$environment/prepare_checkpoint.sh" "$environment" "$environment/cp725.assets.json"
fi
