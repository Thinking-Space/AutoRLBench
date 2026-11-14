#!/usr/bin/env bash
# maestro_grpo_reward_collapse — prepare stage.
#
# Runs in the prepare-stage container: root, the packaged
# vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12 image, the only networked
# stage, and $AUTORLBENCH_WORKSPACE / $AUTORLBENCH_PREPARED writable.
#
# Four blocks, in order:
#   1. numpy 1.26.0 + scipy 1.15.3 shadowed over the image's numpy 2.2.6.
#   2. The GLM service stack rebuilt as a uv-managed CPython 3.11 venv under
#      $AUTORLBENCH_PREPARED/glm (the recorded dedicated GLM environment,
#      verl_glm_ocr / vllm 0.16.0 / torch 2.9.1 / transformers 5.2.0); the
#      image's own vLLM 0.11 stack cannot serve GLM-4.6V, so this second
#      interpreter is what scripts/launch_with_models.py points GLM_PYTHON
#      at. The venv lands under prepared/ (not the workspace) so it rides the
#      stage handoff read-only without bloating the workspace upload.
#   3. Historical lossless-gzip logs restored into the workspace.
#   4. The pinned dataset (19,608 files) and all seven model asset sets,
#      fetched with the task-local fetch_hf.py (revision-keyed, cache-first)
#      and sha256-verified against the manifests by prepare_data.py /
#      prepare_models.py — no host conda and no host model paths.
set -euo pipefail
environment_dir="${AUTORLBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must name the staged workspace}"
prepared="${AUTORLBENCH_PREPARED:?AUTORLBENCH_PREPARED must name the prepared handoff dir}"
export PYTHONDONTWRITEBYTECODE=1

# --- recorded numpy/scipy, shadowed over the image ----------------------------
# The image's compiled layer matches the recorded training stack version-for-
# version (vLLM 0.11.0 / torch 2.8.0 / transformers 4.57.6 / triton 3.4.0 /
# xformers 0.0.32.post1 / flash-attn 2.8.3.post1 / nccl 2.27.3 /
# cudnn 9.10.2.21, all cu128); the deviations are the interpreter (recorded
# py3.10 vs image py3.12), ray 2.54.0 vs the image's 2.56.0, and pure-Python
# pins. Of those only the numpy major is load-bearing: the vendored verl pins
# numpy<2.0.0 (verl/setup.py) while the image froze numpy 2.2.6, and NumPy 2's
# NEP 50 promotion rules can shift mixed-dtype arithmetic during training.
# scipy follows numpy into the shadow (its image build is compiled against
# numpy>=2; the ABI clash would break transformers' lazy imports). --no-deps:
# everything else these two need already sits in the image at
# recorded-compatible versions. The same shadow pattern was verified on this
# image on real AVX-512 x86 (vl-grpo and PiSSA tasks, 2026-11-10/11).
site="$prepared/pysite"
if [ ! -d "$site/numpy" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" numpy==1.26.0 scipy==1.15.3
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env. The explicit /workspace keeps the image's PYTHONPATH ENV
# in play — the prepared/env line replaces it. scripts/project_environment.sh
# prepends the workspace and vendored verl to whatever it inherits.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "$prepared/env"; then
    echo "PYTHONPATH=$site:/workspace" >> "$prepared/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import numpy, scipy
assert numpy.__version__ == "1.26.0", numpy.__version__
assert scipy.__version__ == "1.15.3", scipy.__version__
EOF

# --- GLM service venv (recorded vllm_glm_ocr stack, second interpreter) ------
# GLM-4.6V-Flash is the one expert the image's stack cannot serve: its
# architecture needs the vLLM 0.16 / transformers 5.2 generation, which is why
# the recording rig kept a dedicated conda env. Here that env is replayed as
# a uv-managed CPython 3.11 venv (recorded patch: 3.11.14 — uv installs the
# exact patch when python-build-standalone carries it, else the newest 3.11.x;
# the deviation, if any, is recorded in this comment). The 165-pin freeze in
# glm-requirements.txt is the recorded environment's complete installed set
# (clean: cu12-only, no flash-attn), installed with --no-deps exactly as
# frozen. uv's python-build-standalone download needs GitHub access, same as
# the HF fetches below; the same mechanism is already used by
# infrastructure/trl_tool_history_cross_contamination.
glm_venv="$prepared/glm"
# Guard on a completion stamp rather than the venv's existence: a network
# failure mid-`uv pip install` leaves a created-but-incomplete venv behind
# (the interpreter exists, the packages don't), which an existence check would
# then forever skip. Only a fully installed, asserted venv gets the stamp.
glm_stamp="$glm_venv/.installed"
if [ ! -f "$glm_stamp" ]; then
    rm -rf "$glm_venv"
    _pip_args=(--no-cache-dir)
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" --break-system-packages uv==0.9.9
    export UV_PYTHON_INSTALL_DIR="$prepared/uv-python"
    if ! uv python install 3.11.14; then
        echo '[prepare] cpython 3.11.14 not on python-build-standalone; using the newest 3.11.x' >&2
        uv python install 3.11
    fi
    uv venv --python 3.11 "$glm_venv"
    _uv_args=(--python "$glm_venv/bin/python" --no-deps)
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _uv_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
    fi
    uv pip install "${_uv_args[@]}" -r "$environment_dir/glm-requirements.txt"
    touch "$glm_stamp"
fi
# The venv is self-contained: neither this assert nor any later stage may see
# the cp312-built pysite shadow on PYTHONPATH (scripts/launch_with_models.py
# strips it again around the GLM service subprocess). The script itself
# exported that shadow for the cp312 blocks above, so drop it here.
env -u PYTHONPATH "$glm_venv/bin/python" - <<'EOF'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
import flashinfer, torch, transformers, vllm
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert transformers.__version__ == "5.2.0", transformers.__version__
assert vllm.__version__ == "0.16.0", vllm.__version__
assert flashinfer.__version__ == "0.6.3", flashinfer.__version__
EOF
# GLM_PYTHON rides the prepared/env handoff into the rerun stage; run.py and
# scripts/launch_with_models.py read it from the environment. PYTHON /
# VLLM_PYTHON are pinned to the image interpreter the same way — the recorded
# task declared both statically at /opt/maestro/bin/python, and scripts/
# project_environment.sh refuses to run without one of the two being set.
if ! grep -qs "^GLM_PYTHON=" "$prepared/env"; then
    echo "GLM_PYTHON=$glm_venv/bin/python" >> "$prepared/env"
fi
if ! grep -qs "^PYTHON=" "$prepared/env"; then
    image_python="$(command -v python3)"
    printf 'PYTHON=%s\nVLLM_PYTHON=%s\n' "$image_python" "$image_python" >> "$prepared/env"
fi

# --- recorded logs, dataset, and the seven model asset sets -------------------
python3 "$environment_dir/restore_logs.py" --workspace "$workspace"
python3 "$environment_dir/prepare_data.py" --workspace "$workspace"
python3 "$environment_dir/prepare_models.py" --workspace "$workspace"
printf '%s\n' '[prepare] GLM venv ready; recorded numpy shadowed; historical logs restored; dataset, tool inputs and all seven models verified.'
