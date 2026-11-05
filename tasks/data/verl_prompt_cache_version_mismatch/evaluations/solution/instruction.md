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
repaired workspace as a foreground process:

```bash
bash run.sh
```

with this environment:

```text
WANDB_MODE=offline
```

The effective run must keep:

- model `/workspace/model`, the prepared Qwen3-1.7B checkpoint, with thinking disabled;
- train data `/workspace/data/train.parquet`, the prepared DAPO V2 rows with all original duplicates retained;
- validation data `/workspace/data/val.parquet`, AIME24 and AIME25, 30 questions each;
- **60 trainer/global steps**, with validation before training and every 10 steps, including step 60;
- the staged 2-worker × 4-GPU GRPO topology, batch 512, 8 responses per prompt and PPO mini-batch 32;
- independent correctness scoring, the staged GRPO objective, KL disabled, and the configured optimizer and sampling settings;
- 32 stochastic validation answers per question, temperature 0.6, top-p 0.95 and top-k 20;
- a final HuggingFace checkpoint at step 60; optimizer-state persistence is not required.

The model, prepared data, validation questions/golds and correctness scorer are
fixed. Repair the data-processing behavior while preserving the published
training and evaluation contract. Changing the reward definition, replacing
data, reducing the step budget, altering validation sampling or pointing the
run at another model is not an accepted repair. Do not copy stored historical
scores or answers into the fresh output directory.

The launch script writes the fresh `metrics.jsonl`, `fresh_eval/60.jsonl`,
`effective_config.json`, training log and final checkpoint beneath
`$AUTORLBENCH_RESULTS/experiment_logs`. The verifier reads only that fresh run;
logs elsewhere are ignored. Do not copy, edit or fabricate log metrics.
Otherwise the run is scored 0.

Completion is required independently of metric quality: the complete initial
and every-10-step validation schedule, all training steps 1–60 and the final
step-60 validation must be present. Evaluation beyond the fixed horizon is not
accepted. The final metric is the arithmetic mean of AIME24 and AIME25
mean@32, recomputed from each question's 32 answers; this is not pass@32.

A missing or empty `/workspace/report.md` scores 0. The harness starts the
experiment only after the report is submitted; do not launch training yourself.
