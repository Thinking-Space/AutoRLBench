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
are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process, using the launch script:

```text
examples/retool/retool_qwen3_4b_rl.sh
```

with this environment:

```text
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run must use:

- HF checkpoint `/workspace/font-info_qwen3-4b-sft`
- Megatron torch_dist reference `/workspace/font-info_qwen3-4b-sft_torch_dist`
- train data `/workspace/zhuzilin_dapo-math-17k/dapo-math-17k.jsonl`
- validation data `/workspace/zhuzilin_aime-2024/aime-2024.jsonl`
- `--num-rollout 70`
- `--eval-interval 10`

**Make sure your repaired launch script actually produces that run.** The
staged `examples/retool/retool_qwen3_4b_rl.sh` bakes `--num-rollout 3000`
directly into the training invocation with no environment override, and the
graded rerun passes no budget override either — if you leave 3000 in place,
the rerun blows its time budget and scores 0. Edit the script to the required
70.

Two more traps specific to this task:

- **Log placement is your script's job.** The launch script builds the Ray
  runtime-env JSON itself and only propagates a few variables into the
  training process — `WANDB_DIR` set in the environment does **not** reach it.
  Your repaired script must itself write the training log and the offline
  W&B data under `$AUTORLBENCH_RESULTS/experiment_logs` (offline W&B below
  `$AUTORLBENCH_RESULTS/experiment_logs/wandb`), or the grader will find no
  metrics to grade.
- **The W&B key.** The script passes `--wandb-key ${WANDB_API_KEY}` unquoted;
  offline mode gets a harmless local placeholder from the environment, so the
  script must not choke when it is a placeholder rather than a real key.

A task-specific grading script then reads **only this fresh run's** files
under `$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs
written anywhere else are ignored. Do not copy, edit, or fabricate log
metrics. Otherwise the run is scored 0.
