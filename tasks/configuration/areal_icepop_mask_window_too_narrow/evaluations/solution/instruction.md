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
are fine. **Do not use the network.** Do not fetch code, documentation,
packages, or any other material from outside the staged workspace — diagnose
and repair using only what is staged locally.

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
`$AUTORLBENCH_RESULTS/run`, then re-runs the experiment in your repaired
workspace as a foreground process:

```bash
bash run.sh
```

with this environment:

```text
RLROOT=$AUTORLBENCH_RESULTS/run
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/run
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The effective run uses:

- model `/workspace/models/Qwen1.5-MoE-A2.7B-Chat`
- train and validation data `/workspace/data/gsm8k` (the GSM8K `main` split),
  with the prompt handling and reward logic as shipped
- the budget already in the launch script: `total_train_epochs=3` (87 training
  steps), seed 2, and the allocation plus every other default of
  `examples/math/gsm8k_icepop.yaml` as shipped

`run.sh` launches the full experiment; the held-out evaluation runs at the end
of each of the three epochs. A task-specific grading script then reads **only
this fresh run's** logs under `$AUTORLBENCH_RESULTS/run` and scores the final
held-out evaluation. Logs written anywhere else are ignored, and so are the
frozen historical logs already staged under `run/` — do not copy, edit, or
fabricate log metrics, and do not replay the frozen historical logs as a new
run. Otherwise the run is scored 0.

The rerun must keep the task's data and code: the GSM8K dataset files, the
model directory's configuration and tokenizer files, the framework source
tree (`areal/`), the example entry points (`examples/`), and the project lock
files are pinned by sha256 at grading time, so changing any of them fails the
run, and a run that does not reach its final held-out evaluation fails the
completeness check.
