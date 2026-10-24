#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must point to the staged workspace}"
test -s "$AUTORLBENCH_WORKSPACE/outputs/training.log"
test -s "$AUTORLBENCH_WORKSPACE/outputs/training_metrics.json"
test -s "$AUTORLBENCH_WORKSPACE/outputs/rollout_samples.jsonl"
test -s "$AUTORLBENCH_WORKSPACE/outputs/evaluation_metrics.json"
test -s "$AUTORLBENCH_WORKSPACE/model/generation_config.json"
