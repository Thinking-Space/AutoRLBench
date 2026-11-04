#!/usr/bin/env bash
# trl_gsm8k_eos_mismatch: verify the frozen evidence the issue tier reads.
#
# Runs in the separate prepare-stage container: root, networked, with
# $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS writable.
#
# The diagnosis snapshot deliberately ships no weights (the SFT checkpoint is
# submitter-private) and the offline evaluation needs none. Here we only
# verify that the frozen evidence is staged and non-empty.
set -euo pipefail

: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must point to the staged workspace}"
test -s "$AUTORLBENCH_WORKSPACE/outputs/training.log"
test -s "$AUTORLBENCH_WORKSPACE/outputs/training_metrics.json"
test -s "$AUTORLBENCH_WORKSPACE/outputs/rollout_samples.jsonl"
test -s "$AUTORLBENCH_WORKSPACE/outputs/evaluation_metrics.json"
test -s "$AUTORLBENCH_WORKSPACE/model/generation_config.json"
test -s "$AUTORLBENCH_WORKSPACE/model/added_tokens.json"
