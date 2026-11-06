#!/usr/bin/env bash
# verl_grpo_uid_after_repeat — prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 image (upstream
# verlai/verl:vllm017.latest), **the only stage with network**. In the agent
# stage this file is mounted read-only at /autorlbench/environment.
# Maintainer notes (defect details, grading thresholds) go in ../README.md,
# not here.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# --- Base model -------------------------------------------------------------
# examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh:30-38 searches, in order,
# /models/... -> /workspace/verl/Llama-3.2-1B-instruct -> a placeholder path.
# Nothing mounts the first candidate, so the model must land exactly on the second.
# Mind the case: the directory name is `Llama-3.2-1B-instruct` (lowercase i), which
# differs from the upstream repo name, but that is what the script expects.
#
# (Historical pin, kept as a reference: the old shared-binary fetch pulled
# LIF1014/autorlbench-Llama-3.2-1B-Instruct @
# 32f9e721f796d3c639ab64d2f58ceeaa2c3213b9, model.safetensors sha256
# 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f /
# 2471645608 bytes. That repo is gone from HF; LIF1014/ptdbench-Llama-3.2-1B-Instruct
# is the re-hosted copy, fetched whole-repo by the task-local fetch_hf.py.)
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo LIF1014/ptdbench-Llama-3.2-1B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/verl/Llama-3.2-1B-instruct"

# --- Dataset ----------------------------------------------------------------
# The same script (:44-62) requires /workspace/DAPO-Math-17k_verl/{train,test}.parquet.
# Upstream BytedTsinghua-SIA/DAPO-Math-17k is a single 1,791,700-row shard that
# repeats the same 17,917 problems 100x, so it cannot be fetched as-is: the
# deterministic recipe in autorlbench_datasets/dapo_math_17k.py (pinned upstream
# revision, UUID-keyed 90/10 train/test split) dedupes and splits it. The
# derived parquets are cached under $AUTORLBENCH_DATASETS/derived/ and reused
# across runs.
python3 "$AUTORLBENCH_ENVIRONMENT/autorlbench_dataset.py" \
    --recipe dapo_math_17k \
    --dest "$AUTORLBENCH_WORKSPACE/DAPO-Math-17k_verl"

# The recorded environment ran with PYTHONDONTWRITEBYTECODE=1. It used to be a
# task.toml [environment.env] entry, but that env applies to every stage alike
# (and the offline entries there would have switched the network off for this
# stage too) — seeding the prepared env is the stage-scoped way to keep it.
if ! grep -qs "^PYTHONDONTWRITEBYTECODE=" "$AUTORLBENCH_PREPARED/env"; then
    echo "PYTHONDONTWRITEBYTECODE=1" >> "$AUTORLBENCH_PREPARED/env"
fi
