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

If you find a defect, fix it in the staged workspace at `/workspace`. Then run
the experiment as a foreground process and wait until it returns. Do not use
the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The dataset, held-out set, model, training-step budget, and
evaluation interval are fixed.

The launch script is:

```text
run_calc_grpo.sh
```

Run it from `/workspace` with the log directory set, for example:

```bash
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs"
LOG_DIR="$AUTORLBENCH_RESULTS/experiment_logs" bash run_calc_grpo.sh
```

The effective run must use:

- model `/workspace/models/Qwen3-1.7B`
- train data `/workspace/data/gsm8k/train.parquet`
- validation data `/workspace/data/gsm8k/test.parquet` and `/workspace/data/gsm8k/test_hard.parquet`
- the budget already in the script: `trainer.total_training_steps=100`, `trainer.test_freq=10`,
  `data.train_batch_size=64`, `actor_rollout_ref.rollout.n=8`,
  `actor_rollout_ref.rollout.calculate_log_probs=True`, do not change these
- `LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs`

Create `$AUTORLBENCH_RESULTS/experiment_logs` first. Offline W&B data must be
created below `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. The run is scored
only from files under `$AUTORLBENCH_RESULTS/experiment_logs`. Logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics.
Otherwise the run is scored 0.
