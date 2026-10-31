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
mechanism. The system runs the experiment itself after you finish (see below).
anything you launch is wasted and scores nothing. Cheap, non-training checks
are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause**: the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes**: every file you changed and what you changed in it.
3. **Expected effect**: which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_gsm8k_grpo.sh
```

with this environment:

```text
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses:

- model `/workspace/models/Qwen2.5-0.5B-Instruct`
- train data `/workspace/data/gsm8k/train.parquet`
- validation data `/workspace/data/gsm8k/test.parquet`, with the prompt as shipped
  (it asks for the final answer after `####`)
- the budget already in the script: `trainer.total_training_steps=100`, `trainer.test_freq=10`,
  `data.train_batch_size=128`, `actor_rollout_ref.rollout.n=8`,
  `actor_rollout_ref.rollout.val_kwargs.do_sample=False`, `data.max_response_length=512`
- `trainer.validation_data_dir=$LOG_DIR/val_generations` as shipped: the grader scores the
  held-out generations the run dumps at step 100, not a logged number

The grader re-scores `experiment_logs/val_generations/100.jsonl` itself, with a fixed rule that
is not part of the workspace: a response is correct only if the last `#### <number>` it contains
equals the gold answer numerically. Changing the logged metric, the reward manager, the
validation set, the prompt, or the dump path does not change that rule. A run whose step-100 dump
is missing scores 0.

A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics. Logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics or
generation dumps, and do not replay the frozen historical logs under
`results/` as a new run. Otherwise the run is scored 0.
