#!/usr/bin/env bash
# staged task: verify the frozen evidence the issue tier reads.
#
# Runs in the separate prepare-stage container: root, networked, with
# $AUTORLBENCH_WORKSPACE writable. The task ships no weights and the offline
# diagnosis needs none, so this stage only checks that the snapshot is staged
# and non-empty.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must point to the staged workspace}"
W="$AUTORLBENCH_WORKSPACE"

test -s "$W/outputs/metrics.json"
test -s "$W/outputs/historical_train_trace.log"
test -s "$W/outputs/launch_command.txt"

test -s "$W/miles/rollout/rm_hub/llm_judge.py"
test -s "$W/model/chat_template.jinja"
test -s "$W/rl_data/chimera_judge_gt_equiv_reasonfirst.txt"
test -s "$W/scripts/minicpm5/2_6b/short_rl/run-minicpm5-26b-short-math-8k-colocate.sh"
