#!/usr/bin/env bash
# ppo_critic/aime25_single_sample_variance: what to prepare before training.
#
# Runs in a separate prepare-stage container: root, environment activated, **the only stage with network**,
# $AUTORLBENCH_WORKSPACE / $AUTORLBENCH_PREPARED / $AUTORLBENCH_MODELS / $AUTORLBENCH_DATASETS are all writable.
#
# This task is the special case among the four ppo_critic tasks: it changes the **evaluation protocol**
# (samples_per_question in config/eval_config.yaml), and grading neither trains nor runs a real
# evaluation, so the prepare stage **has nothing to download**. This is deliberate, not an omission. Reasons:
#
#   * Model: not needed. Neither grading level loads weights:
#       - solution/verifier only runs `run_eval.py --protocol-only` (run_eval.py:93-97:
#         prints the protocol JSON only, loads no checkpoint, starts no vLLM) plus
#         `sha256sum -c` over the original_eval/ evidence; both steps use only the Python stdlib.
#       - issue/verifier is an LLM judge that compares text and never touches a model.
#   * Datasets/evidence: nothing to download. All controlled repeated-sampling evidence ships
#     with the package and is checked during grading:
#       workspace/autorlbench_case/original_eval/{n1,n32,notes,SHA256SUMS.txt}.
#   * The optional "full AIME25 rerun" needs an **external PPO-Critic checkpoint**, passed via the
#     CHECKPOINT_PATH environment variable (run_eval.py:116-161 requires config.json and
#     model.safetensors.index.json in the checkpoint directory). That is a large trained checkpoint, not
#     an HF base model, so there is no repo for autorlbench_fetch_hf; whether to provide it is up to
#     the operator, and grading for this task does not depend on it.
#
# Conclusion: the prepare stage is a documented no-op. This file is kept so the environment/ layout
# matches the other three ppo_critic tasks, and so "why nothing is downloaded" is written where it belongs.
set -euo pipefail

echo "[prepare] aime25_single_sample_variance: nothing to fetch."
echo "[prepare]   grading is stdlib-only (run_eval.py --protocol-only + sha256sum -c);"
echo "[prepare]   preserved evidence ships in autorlbench_case/original_eval/;"
echo "[prepare]   the optional full rerun takes an external checkpoint via CHECKPOINT_PATH."
