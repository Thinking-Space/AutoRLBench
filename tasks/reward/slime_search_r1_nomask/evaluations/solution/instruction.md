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

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.

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
examples/search-r1/run_qwen2.5_3B.sh
```

with this environment:

```text
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run must use:

- HF checkpoint `/workspace/Qwen2.5-3B`
- Megatron torch_dist reference `/workspace/Qwen2.5-3B_torch_dist`
- train data `/workspace/nq_hotpotqa_train/train.parquet`
- validation data `/workspace/nq_hotpotqa_train/test.parquet`
- `--num-rollout 100`
- `--eval-interval 10`
- local retrieval server `http://127.0.0.1:8000/retrieve`

**Make sure your repaired launch script actually produces that run.** The
staged `examples/search-r1/run_qwen2.5_3B.sh` bakes `--num-rollout 200`
directly into the training invocation with no environment override, and the
graded rerun passes no budget override either — if you leave 200 in place,
the rerun blows its time budget and scores 0. Edit the script to the required
100.

Four more traps specific to this task:

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
- **Read-only scoring assets.** `examples/search-r1/qa_em_format.py` (the
  exact-match scorer the launch script wires as the custom reward manager),
  `examples/search-r1/local_dense_retriever/` (the retrieval server), and the
  staged datasets under `nq_hotpotqa_train/` are verified correct and graded
  read-only: after the rerun the verifier hashes them, and any modification,
  replacement, or deletion scores 0 no matter what the metrics say. The
  `--num-rollout` fix and the loss-mask repair belong in
  `examples/search-r1/generate_with_search.py` and
  `examples/search-r1/run_qwen2.5_3B.sh`, not in the scorer or the data.
  A quirk that looks like a defect in these files (for example, how
  `extract_solution` treats a single `<answer>` tag) is part of the verified
  recipe and is not yours to change.
- **The launch script's scoring and eval wiring is fixed.** Beyond the
  instructed `--num-rollout 100` edit and the log-placement repair, the script
  must keep wiring the run exactly as staged: the custom generate/reward-manager
  paths (`generate_with_search.generate` /
  `generate_with_search.reward_func`), the eval on `nq_test` over the staged
  `test.parquet` with its `[0:3000]` slice, `--eval-interval 10`, and
  `--n-samples-per-eval-prompt 1`. The verifier checks these after the rerun;
  repointing the scorer, swapping the eval dataset, or resampling evals scores
  0 no matter what the metrics say.

A task-specific grading script then reads **only this fresh run's** files
under `$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs
written anywhere else are ignored. Do not copy, edit, or fabricate log
metrics. Otherwise the run is scored 0.

Completion is required independently of metric quality: validation at the
declared final step must be present, and evaluation beyond the fixed horizon
is not accepted. Where training reward is graded, every training step must be
logged. Accuracy and binary rewards must remain in their valid [0, 1] range.
