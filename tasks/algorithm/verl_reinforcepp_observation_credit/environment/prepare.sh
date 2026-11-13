#!/usr/bin/env bash
# verl_reinforcepp_observation_credit — prepare stage.
#
# Runs in the prepare-stage container: root, the verl-vllm0.24 image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It shadows the
# recorded pure-Python pins, fetches the Qwen3-4B weights, pip-installs the
# vendored verl checkout, byte-compiles the reproductions package and
# generates the operations-SQL workload.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"

# --- recorded pure-Python pins, shadowed over the image ----------------------
# The task runs on the packaged verl-vllm0.24 image (torch 2.11.0+cu130 /
# vLLM 0.24.0 / flashinfer 0.6.12 / py3.12): the compiled layer matches the
# recorded tool-rl stack, including the actively-used engine (rollout.name=
# vllm, mode=async — vLLM 0.24.0 is the recorded version, so the vendored
# verl 0.9.0.dev talks to the same engine API). The image's pure-Python
# layer is frozen at different versions, so the recorded majors are
# shadowed from $AUTORLBENCH_PREPARED/pysite; tokenizers (0.22.2) already
# matches the image. --no-deps: the image carries the transitive closure,
# and letting pip resolve would float pins the image deliberately holds.
# Native drift the shadow cannot reach (ray 2.56.1 vs 2.60.0, nccl 2.30.7
# vs 2.28.9, pyarrow 25.0.0 vs 26.0.1, numpy 1.26.4 vs 2.3.5) is accepted —
# the defective run was pilot-verified on this image (2026-11-13); the full
# reference-evidence re-record on this image is pending.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/transformers" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        transformers==5.10.1 accelerate==1.15.0 datasets==5.1.0 peft==0.21.2 \
        wandb==0.31.0 huggingface_hub==1.33.0
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env (run.sh re-prefixes it ahead of $PWD/vendor/verl).
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "PYTHONPATH=$site:$W/vendor/verl:$W" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import transformers, accelerate, datasets, peft, wandb, huggingface_hub
assert transformers.__version__ == "5.10.1", transformers.__version__
assert accelerate.__version__ == "1.15.0", accelerate.__version__
assert datasets.__version__ == "5.1.0", datasets.__version__
assert peft.__version__ == "0.21.2", peft.__version__
assert wandb.__version__ == "0.31.0", wandb.__version__
assert huggingface_hub.__version__ == "1.33.0", huggingface_hub.__version__
EOF

# --- image-level env, seeded here --------------------------------------------
# The verl-vllm0.24 image bakes no CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM
# defaults; the two entries the recording rig's image carried ride the
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

python3 -m pip install --no-deps --no-build-isolation --break-system-packages -e "$W/vendor/verl"
python3 -m compileall -q "$W/reproductions"
cd "$W"
python3 -m reproductions.sql_credit.workload --output data/sql
