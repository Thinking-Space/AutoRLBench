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
repaired workspace as a foreground process:

```bash
bash run_grpo_lora.sh data.train_files=$PWD/data/gsm8k/train.parquet data.val_files=$PWD/data/gsm8k/test.parquet data.train_batch_size=128 data.max_response_length=1024 actor_rollout_ref.model.path=$PWD/models/Qwen2.5-1.5B-Instruct actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 actor_rollout_ref.model.lora.merge=True actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1 reward.custom_reward_function.path=null trainer.n_gpus_per_node=1 trainer.total_training_steps=60 trainer.test_freq=5 trainer.val_before_train=True trainer.validation_data_dir=$LOG_DIR/val_generations
```

with this environment:

```text
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The overrides are appended after the launch script's own, so they take effect
whatever the script says. The effective run uses:

- model `/workspace/models/Qwen2.5-1.5B-Instruct`
- train data `/workspace/data/gsm8k/train.parquet` (7473 GSM8K training rows)
- validation data `/workspace/data/gsm8k/test.parquet` (1319 GSM8K test rows)
- 60 training steps, validation before training and every 5 steps
- LoRA rank 32, alpha 64, merged into the rollout weights, FSDP2 actor, one GPU
- `WANDB_MODE=offline`

The launcher writes stdout to `experiment_logs/train.log`, offline W&B data
under `experiment_logs/wandb`, and validation generations under
`experiment_logs/val_generations`. A task-specific grading script then reads
**only this fresh run's** files under `$AUTORLBENCH_RESULTS/experiment_logs`
and scores the key metrics; logs written anywhere else are ignored. Keep the
model, data, reward function and its scorer, LoRA configuration, batch sizes,
training budget and evaluation schedule unchanged, and do not turn off LoRA
merging or FSDP2 to avoid an implementation defect. Do not copy, edit, or
fabricate metrics or replay frozen historical logs as a new run. Otherwise the
run is scored 0. Leave the frozen evidence under `outputs/` intact.
