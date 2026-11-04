# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use the rollout, throughput, recycling, truncation, runtime, optimization, and
training signals together. This is a fully-async run trained against a fixed
wall-clock budget rather than a fixed step count, so how much useful work the
engine completes per unit time is itself part of health.

If you find a defect, fix it in the staged workspace at `/workspace`. Do not
use the network. The dataset, model, rollout configuration, and wall-clock
training budget are fixed.

**Do not launch the experiment — do not run `scripts/train_grpo.sh` yourself,
in the foreground or background, and do not use `nohup`, `&`, `tmux`,
`screen`, or any detach mechanism.** The system runs the full 8-hour
experiment itself after you finish; a run you start is wasted and scores
nothing.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Diagnosis** — the defect you found and the evidence that connects it to
   the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — the metrics the fix should move and their expected
   values.

A missing or empty `/workspace/report.md` scores 0.

## How you will be checked

After you submit the report, the system runs the experiment end-to-end in
your repaired workspace via the launch script

```text
scripts/train_grpo.sh
```

with the effective run fixed to:

- model `/workspace/Qwen3-1.7B`
- Megatron torch_dist reference `/workspace/Qwen3-1.7B_torch_dist`
- train data `/workspace/data/dapo_math_17k_dedup.jsonl`
- `TRAIN_SECONDS=28800` (the fixed 8-hour wall-clock budget)
- logs under `$AUTORLBENCH_RESULTS/experiment_logs`

The run is scored only from files under `$AUTORLBENCH_RESULTS/experiment_logs`
(the driver log and the per-step rollout dumps `dump/rollout_data/*.pt`); logs
written anywhere else are ignored. Do not copy, edit, or fabricate log metrics
or dumps. Otherwise the run is scored 0.

Scoring reads two signals from those logs: the number of training steps
completed in the fixed wall clock (primary), and the last-window mean solve
rate over the dumped rollouts (secondary). Both must clear their thresholds.
The fixed launch configuration (model, dataset, rollout settings, wall-clock
budget) is itself machine-checked from the run's own logs: a run that relaxes
any of it instead of repairing the defect scores 0.
