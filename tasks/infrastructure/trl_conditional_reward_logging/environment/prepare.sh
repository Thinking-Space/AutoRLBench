#!/usr/bin/env bash
# Prepare stage, shared byte-identical by the trl-grpo-torch26 task pair
# (trl_dapo_accumulation_normalization, trl_conditional_reward_logging).
#
# Runs in the prepare-stage container: root, the trl-grpo-torch26 image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It fetches the
# Qwen2.5-0.5B-Instruct weights and the GSM8K parquet, then pip-installs the
# vendored TRL checkout from the workspace -- the defective tree the agent
# repairs. Destinations are what run.py's flags read: --model model --data data.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- image-level env, seeded here --------------------------------------------
# docker/images/trl-grpo-torch26.Dockerfile is a build recipe baked without
# ENV; the two entries the recording rig's image carried ride the
# prepared/env handoff into the agent/rerun/verifier stages via the activate
# hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- Base model ---------------------------------------------------------------
# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions the former autorlbench_fetch_hf
# allowlist pinned are recorded here for reference:
#   Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
#     (main/train-00000-of-00001.parquet, main/test-00000-of-00001.parquet)
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --dest "$W/model"

# --- Dataset ------------------------------------------------------------------
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data"
ln -sf main/train-00000-of-00001.parquet "$W/data/train.parquet"
ln -sf main/test-00000-of-00001.parquet "$W/data/test.parquet"

# --- Vendored TRL -------------------------------------------------------------
python3 -m pip install --no-deps --no-build-isolation "$W"
