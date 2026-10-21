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
scripts/train.sh
```

Run it as a foreground process:

```bash
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs"
TRAINING_RUN_DIR="$AUTORLBENCH_RESULTS/experiment_logs" bash scripts/train.sh
```

The effective run must use:

- model `/workspace/models/Qwen1.5-MoE-A2.7B-Chat`
- train data `/workspace/data/gsm8k/train.parquet` (7473 official training rows)
- validation data `/workspace/data/gsm8k/validation64.parquet` (the prepared 64 held-out rows)
- 150 training steps, validation at step 100 and the final step 150
- the supplied configuration, seeds, CP2/EP2/TP1 layout and two allocated GPUs
- `TRAINING_RUN_DIR=$AUTORLBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`

The launcher places stdout under `experiment_logs/logs/train.log` and offline
W&B data under `experiment_logs/wandb`. The final held-out validation must be
logged at step 150. A step-0 score or an earlier best score is not sufficient.
Keep the dataset, validation selection, reward function, model, batch sizes,
training budget and evaluation schedule unchanged. Do not disable configured
loss terms or parallelism to avoid an implementation defect.

Only fresh logs below `$AUTORLBENCH_RESULTS/experiment_logs` are scored. Do not
copy, edit, or fabricate metrics or replay frozen historical logs as a new run.
Otherwise the run is scored 0. Leave frozen evidence under `observed_run/` intact.
