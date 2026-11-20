#!/usr/bin/env bash
# staged task: verify the frozen evidence the issue tier reads.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must point to the staged workspace}"
W="$AUTORLBENCH_WORKSPACE"

test -s "$W/outputs/train.20260715.log"
test -s "$W/outputs/launch_command.txt"
test -s "$W/miles/backends/training_utils/loss.py"
test -s "$W/rl_data/chimera_judge_template_verdict_first.txt"
test -s "$W/scripts/minicpm5/2_6b/short_rl/run-minicpm5-26b-short-math-8k-colocate.sh"
