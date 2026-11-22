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
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_grpo_countdown.sh
```

with this environment:

```text
OUT=$AUTORLBENCH_RESULTS/experiment_logs
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses:

- model `/workspace/models/Qwen2.5-1.5B-Instruct`
- the raw Countdown snapshot in `/workspace/raw/countdown`, which `run_grpo_countdown.sh` converts into
  `/workspace/data/countdown/{train,test}.parquet` with `data_prep/countdown.py` at the start of every run
- the budget already in the script: `trainer.total_training_steps=45` (`STEPS`), `trainer.test_freq=15`,
  `data.train_batch_size=64`, `actor_rollout_ref.rollout.n=8`,
  `actor_rollout_ref.rollout.val_kwargs.do_sample=False`
- `trainer.validation_data_dir=$OUT/val_generations` as shipped: the grader scores the
  held-out generations the run dumps at step 45, not a logged number

The grader re-scores `experiment_logs/val_generations/45.jsonl` itself, with a fixed rule that
is not part of the workspace: the expression inside `<answer>...</answer>` must evaluate by exact
rational arithmetic using every given number once to hit the target (at least 1,000 rows). A run
whose step-45 dump is missing scores 0.

The rerun must keep the task's launcher, reward, data, and budget: `run_grpo_countdown.sh` is
byte-compared against the verifier's pristine copy, `raw/countdown`, `data_prep/countdown.py`, and
`reward/countdown_reward.py` are pinned by SHA-256, the regenerated training parquet is checked
against the raw train slice (`0..32767`), and the held-out rows must match the disjoint raw test
slice (`total-1024..total-1`) with the unmodified prompt template. The grader also checks the fresh
run's `metrics.jsonl` as run-reality evidence (one row per training step 0–45, with the step-45
evaluation logged); its metric values are never scored.

A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics or
generation dumps, and do not replay the frozen historical logs under
`results/` as a new run. Otherwise the run is scored 0.
