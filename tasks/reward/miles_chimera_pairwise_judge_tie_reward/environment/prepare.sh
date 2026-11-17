#!/usr/bin/env bash
# miles_chimera_pairwise_judge_tie_reward: verify the frozen evidence the issue tier reads.
#
# Runs in the separate prepare-stage container: root, networked, with
# $AUTORLBENCH_WORKSPACE writable. The task ships no weights and the offline
# diagnosis needs none, so this stage only checks that the snapshot is staged
# and non-empty.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE must point to the staged workspace}"
W="$AUTORLBENCH_WORKSPACE"

# Per-step trace, the sanitized job entry, and the launch/verifier record the oracle cites.
test -s "$W/outputs/metrics.json"
test -s "$W/outputs/historical_train_trace.log"
test -s "$W/outputs/launch_command.txt"

# The defect itself: the pairwise score table, the A/B template, and the reward path.
test -s "$W/miles/rollout/rm_hub/llm_judge.py"
test -s "$W/rl_data/chimera_judge_template_verdict_first.txt"

# The launch script named by training.script must exist in the workspace.
test -s "$W/scripts/minicpm5/2_6b/short_rl/run-minicpm5-26b-short-math-8k-colocate.sh"
