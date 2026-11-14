# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system runs the experiment itself after you finish (see below);
anything you launch is wasted and scores nothing. Cheap, non-training checks
are fine (executing candidate solutions against a problem's own tests is a
cheap check, not training).

**Do not use the network.** Do not fetch code, documentation, packages, or any
other material from outside the staged workspace — diagnose and repair using
only what is staged locally.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause**: the defect you found, and the evidence in the workspace
   (logs, metrics, code, data) that connects it to the observed behavior.
2. **Changes**: every file you changed and what you changed in it.
3. **Expected effect**: which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
bash examples/grpo_trainer/run_qwen3_4b_instruct_2507_taco_fsdp.sh
```

with this environment:

```text
MODEL_PATH=$AUTORLBENCH_WORKSPACE/Qwen3-4B-Instruct-2507
TRAIN_FILE=$AUTORLBENCH_WORKSPACE/taco_verl/train.parquet
TEST_FILE=$AUTORLBENCH_WORKSPACE/taco_verl/test.parquet
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
ROLLOUT_DATA_DIR=$AUTORLBENCH_RESULTS/experiment_logs/rollout_data
VALIDATION_DATA_DIR=$AUTORLBENCH_RESULTS/experiment_logs/validation_data
NGPUS_PER_NODE=4
NNODES=1
TOTAL_TRAINING_STEPS=40
TEST_FREQ=5
WANDB_MODE=offline
```

The effective run uses the budget already in the script (40 steps, 16
prompts x 8 rollouts, `test_freq=5`) and the held-out validation set exactly
as shipped.

The grader re-scores the fresh run's training rollout dumps itself, with a
rule that is not part of the workspace: it independently re-executes each
dumped response and checks whether the run's own reward scoring of it holds
up. A response the run scored 0 that independent re-scoring shows to be
correct is a "correct solution scored 0" sample. Changing the logged metrics,
the reward manager, or the validation set does not change that rule. A run
whose rollout dumps are missing scores 0.

The rerun must use the workspace as shipped apart from your repair: the launch
script and `verl/utils/reward_score/taco.py` are byte-compared against the
verifier's pristine copies at grading time, and `taco_verl/test.parquet` is
pinned by sha256, so editing the runner, the reward, shortening the run,
touching the held-out set, or training on it (train/test prompt overlap is
checked) fails the run. The repaired `train.parquet` must keep the shipped
schema and prompt template on every row, keep at least 1,200 distinct rows,
and stay disjoint from the held-out prompts; its row content is checked for
tampering at grading time, so rewriting prompts or problem statements to hide
training rows from the grader fails the run. A non-empty
`/workspace/report.md` is required. The grader also requires run-reality
evidence: the per-step rollout dumps the trainer wrote for the fresh run,
with their step and score fields, are checked to exist and be internally
consistent; their values are never scored directly.

A task-specific grading script then reads **only this fresh run's** files
under `$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics. Logs
written anywhere else are ignored. Do not copy, edit, or fabricate log metrics
or generation dumps, and do not replay the frozen historical logs as a new
run. Otherwise the run is scored 0.
