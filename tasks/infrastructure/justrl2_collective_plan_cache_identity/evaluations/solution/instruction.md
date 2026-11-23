# Diagnose and repair the staged experiment

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
are fine.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Verdict** — healthy or unhealthy, with the evidence that decides it.
2. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
3. **Changes** — every file you changed and what you changed in it.
4. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_justrl2_ppo.sh
```

with this environment:

```text
OUT=$AUTORLBENCH_RESULTS/experiment_logs
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses:

- HF checkpoint `/workspace/models/JustRL-II-base-model` and Megatron `torch_dist`
  reference `/workspace/models/JustRL-II-base-model-torch_dist`
- the raw Countdown snapshot in `/workspace/raw/countdown` prepared into
  `/workspace/data/countdown/{train,test}.jsonl` (`8,192` train / `1,024` disjoint test)
  with `data_prep/countdown.py`
- the budget already in `run_justrl2_ppo.sh`: 1 node × 8 GPUs colocated (`actor=4`, `critic=4`),
  `--context-parallel-size 2`, `--num-rollout 30` (`STEPS=30`), `--eval-interval 10`,
  `--num-critic-only-steps 3`, `--rollout-batch-size 16`, `--n-samples-per-prompt 4`,
  `--micro-batch-size 4`, `--advantage-estimator ppo`, `--custom-rm-path reward.countdown_reward.compute_reward`
- `MILES_METRICS_FILE=$OUT/metrics.jsonl` and `MILES_EVAL_DUMP_DIR=$OUT/val_generations`
  as shipped: the grader scores the held-out generations dumped at step 29, not a
  logged scalar

The grader re-scores `experiment_logs/val_generations/29.jsonl` itself using exact
rational AST arithmetic over the given Countdown numbers (`>= 1,000` rows). A run
whose step-29 dump is missing scores 0.

The rerun must keep the task's launcher, reward, data, and budget: `run_justrl2_ppo.sh`
is byte-compared against the verifier's pristine copy, `raw/countdown/data/train-00000-of-00001.parquet`,
`data_prep/countdown.py`, and `reward/countdown_reward.py` are pinned by SHA-256, and
`data/countdown/{train,test}.jsonl` are checked against the raw Countdown slices with
the unmodified prompt template. The grader also checks the fresh run's `metrics.jsonl`
for continuous `rollout/step` `0..29`, at least 27 actor updates with `train/pg_loss`,
and the step-29 evaluation row (`eval/countdown`); its metric values are never scored.

A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics or
generation dumps, and do not replay the frozen historical logs under
`results/` as a new run. Otherwise the run is scored 0.
