# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and recorded rollout trajectories. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Base the diagnosis on training reward anomalies and rollout trajectories,
including action formatting, tool interactions, termination, and truncation.
Connect these observations to the source code and effective configuration.
A reward drop or truncated log alone is not proof of a particular root cause.

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, model, training-step budget, and
evaluation interval are fixed.

The launch script is:

```text
run.py
```

Run `python run.py`. This invokes the historical `scripts/train_4_gpus.sh`
with every log, W&B, rollout and checkpoint destination rooted below
`$AUTORLBENCH_RESULTS/experiment_logs`, and disables checkpoint resume. The
effective run must use:

- model `/workspace/models/Qwen3-VL-4B-Thinking`
- the six staged tool-service models under `/workspace/models/`
- train data `/workspace/data/route/train_with_zwz_modified_prompt.parquet`
- validation data `/workspace/data/route/test_subset_100_seed42.parquet`
- `trainer.total_training_steps=200`
- `trainer.test_freq=40`
- `LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs`

Create `$AUTORLBENCH_RESULTS/experiment_logs` first. Offline W&B data must be
created below `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. The run is scored only
from files under `$AUTORLBENCH_RESULTS/experiment_logs`; logs written anywhere
else are ignored. Do not copy, edit, or fabricate log metrics. Otherwise the
run is scored 0.

This case scores recovery from training collapse using the mean training score
at step 200. Do not change the reward function, the datasets, the model, or the
training budget.
