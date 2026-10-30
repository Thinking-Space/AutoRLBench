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

After the report is submitted, the system clears `$AUTORLBENCH_RESULTS/experiment_logs` and runs:

```bash
bash "$AUTORLBENCH_VERIFIER/run_training.sh"
```

This protected runner imports the repaired TRL source from `/workspace/src`. It uses the same configuration as `run.py` and `train.sh`:
Qwen2.5-0.5B-Instruct from `/workspace/model`, 256 GSM8K training questions and 64 held-out questions from `/workspace/data`,
seed 42, 1 GPU process(es), 24 optimizer steps, four completions per prompt, a 256-token completion limit,
gradient accumulation of two microbatches, and four microbatches per generation batch. W&B runs offline.
The wall-clock limit is 1,800 seconds. Model weights, reward logic, data selection and the run budget are fixed by the runner.
Repair the trainer implementation. Changes to the workspace runner do not replace the protected experiment.

The verifier requires completed training, at least one nonzero gradient update, and the final held-out evaluation.
For every optimizer step, the verifier recomputes the DAPO objective from the recorded per-token probabilities, advantages and completion masks. Both microbatches must agree with the accumulation-window objective within relative tolerance 0.0005 and absolute tolerance 0.000002. All 24 optimizer steps must pass, with 48 records and at least one nonzero objective comparison.
This task evaluates optimization correctness and successful execution. Held-out correctness is reported for context and is not a calibrated performance threshold.
Only files from this fresh run are graded. Do not copy, edit or fabricate metrics.
