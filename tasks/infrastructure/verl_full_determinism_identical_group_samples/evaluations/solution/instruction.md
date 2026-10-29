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
repaired workspace as a foreground process: the 30-step run, and after it the
same command once more for 3 steps under `experiment_logs/repeat`:

```bash
bash run_grpo_fd.sh algorithm.adv_estimator=grpo data.train_files=$PWD/data/gsm8k/train.parquet data.val_files=$PWD/data/gsm8k/test.parquet data.train_batch_size=64 data.max_response_length=1024 data.seed=1 actor_rollout_ref.model.path=$PWD/models/Qwen2.5-1.5B-Instruct actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.fsdp_config.full_determinism=True actor_rollout_ref.ref.fsdp_config.full_determinism=True actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.full_determinism=True actor_rollout_ref.rollout.seed=42 ++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1 reward.custom_reward_function.path=null trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.test_freq=5 trainer.val_before_train=True trainer.total_training_steps=30 trainer.rollout_data_dir=$LOG_DIR/rollouts trainer.validation_data_dir=$LOG_DIR/val_generations \
  && LOG_DIR=$LOG_DIR/repeat bash run_grpo_fd.sh <the same overrides> trainer.total_training_steps=3 trainer.rollout_data_dir=$LOG_DIR/repeat/rollouts trainer.validation_data_dir=$LOG_DIR/repeat/val_generations
```

with this environment:

```text
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The overrides are appended after the launch script's own, so they take effect
whatever the script says. The effective run uses:

- model `/workspace/models/Qwen2.5-1.5B-Instruct`
- train data `/workspace/data/gsm8k/train.parquet` (7473 GSM8K training rows), data seed 1
- validation data `/workspace/data/gsm8k/test.parquet` (1319 GSM8K test rows)
- GRPO, 64 prompts × 8 samples per step at temperature 1.0, 30 training steps,
  validation before training and every 5 steps
- full determinism for the rollout, actor and reference, rollout seed 42,
  one GPU and one vLLM replica
- `WANDB_MODE=offline`

The launcher writes stdout to `experiment_logs/train.log`, offline W&B data
under `experiment_logs/wandb`, training rollouts under
`experiment_logs/rollouts` and validation generations under
`experiment_logs/val_generations` (and the same under `experiment_logs/repeat`
for the 3-step repeat). A task-specific grading script then reads **only these
fresh runs'** files under `$AUTORLBENCH_RESULTS/experiment_logs` and scores the
key metrics; logs written anywhere else are ignored. Because the run asks for
full determinism, the repeat must reproduce the main run's first 3 steps.
Keep the model, data, reward function and its scorer,
GRPO settings, batch sizes, sampling settings, training budget and evaluation
schedule unchanged, and do not turn off full determinism or add rollout
replicas to avoid an implementation defect. Do not copy, edit, or fabricate
metrics or rollouts or replay frozen historical logs as a new run. Otherwise
the run is scored 0. Leave the frozen evidence under `outputs/` intact.
