#!/usr/bin/env bash
# verl_reinforcepp_observation_credit — prepare stage.
#
# Runs in the prepare-stage container: root, the tool-rl-vllm024 image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It fetches the
# Qwen3-4B weights, pip-installs the vendored verl checkout, byte-compiles
# the reproductions package and generates the operations-SQL workload.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"

# --- image-level env, seeded here --------------------------------------------
# docker/images/tool-rl-vllm024.Dockerfile is a build recipe baked without
# ENV; the two entries the recording rig's image carried ride the
# prepared/env handoff into the agent/rerun/verifier stages via the activate
# hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repo, cache-first, without
# checksum verification; the revision the former autorlbench_fetch_hf
# allowlist pinned is recorded here for reference:
#   Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3-4B \
    --dest "$W/assets/model"

python -m pip install --no-deps --no-build-isolation -e "$W/vendor/verl"
python -m compileall -q "$W/reproductions"
cd "$W"
python -m reproductions.sql_credit.workload --output data/sql
