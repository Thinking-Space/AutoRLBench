#!/usr/bin/env bash
# Bounded scoring replay for the r1d15 math GRPO experiment.
#
# Executed by the harness verifier stage as the run_spec command (cwd
# /workspace, AUTORLBENCH_VERIFIER pointing at the staged verifier tree) and by
# the 8-GPU BATCH replay entry with the same environment. One script, so the
# two paths cannot drift.
#
# What it does, in order:
#   1. Restore the experiment's launch and instrumentation pipeline (the
#      scripts that start it and the code that records its measurements) from
#      the verifier's pristine copies. Edits to those files in the agent's
#      workspace are neutralised before the run, so the recorded metrics are
#      always produced by the staged code.
#   2. Delete any previous output run directory, so resume cannot pick up a
#      partial state and only this fresh run's records can be graded.
#   3. Run the experiment once with a bounded schedule: 30 training steps,
#      no validation passes, no checkpoints. This is the scoring run; its
#      records under runs/run_001 are the only ones the grader reads.
set -euo pipefail

: "${AUTORLBENCH_VERIFIER:?must point at the staged verifier tree}"
W="$(pwd)"
case "$W" in */workspace) ;; *) echo "replay.sh: expected cwd /workspace, got $W" >&2; exit 64 ;; esac
P="$AUTORLBENCH_VERIFIER/resources/pristine"

for rel in scripts/train.sh scripts/train_entry.py scripts/run_support.py \
           verl/verl/trainer/ppo/ray_trainer.py verl/verl/utils/tracking.py; do
  [ -f "$P/$rel" ] || { echo "replay.sh: pristine copy missing: $P/$rel" >&2; exit 65; }
  mkdir -p "$W/$(dirname "$rel")"
  cp "$P/$rel" "$W/$rel"
done

# A fresh run directory: the frozen historical records under logs/ are shipped
# evidence, never touched; anything the agent may have left under runs/ is
# discarded so the scoring run starts from the staged initial state.
rm -rf "$W/runs/run_001"

exec bash "$W/scripts/train.sh" \
  trainer.total_training_steps=30 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False
