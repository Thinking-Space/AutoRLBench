# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the recorded
workspace and the control run that fixed it.

Expected verdict: **unhealthy**.

## Root cause

- `verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager.load_checkpoint`
  (reached from the trainer's resume path with `trainer.resume_mode=resume_path`)
  - kind: `framework` / `checkpoint-resume`
  - observed: stage 2 of `run_resume.sh` resumes from
    `stage1/checkpoints/global_step_2` with `actor_rollout_ref.actor.optim.lr=2e-6`.
    The engine (`verl/workers/engine/fsdp/transformer_impl.py`) first builds the
    AdamW optimizer and the constant-with-warmup `LambdaLR` scheduler from that
    config (base lr 2e-6), then `load_checkpoint` calls
    `self.optimizer.load_state_dict(...)` (restores every param group's `lr` and
    `initial_lr` from the checkpoint, 1e-6) and
    `self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])`
    (restores `base_lrs = [1e-06]` and `_last_lr`). From then on each
    `lr_scheduler.step()` recomputes `lr = base_lr * lambda(step) = 1e-6 * 1.0`.
    The configured value only exists for the few calls before the checkpoint is
    loaded, and nothing logs or warns about the mismatch.
  - expected: on resume, either keep the configured learning rate (re-apply the
    configured `initial_lr`/`base_lrs` after loading and recompute the group
    `lr`), or refuse/warn when the configured lr differs from the checkpoint. A
    workaround outside the framework is to edit `base_lrs` and the param-group
    `lr`/`initial_lr` in the checkpoint's `extra_state_*` / `optim_*` files
    before resuming, or to start a fresh run instead of resuming.

A passing diagnosis must state that the stage-2 run actually trains at 1e-6, not
the configured 2e-6, **and** attribute this to checkpoint resume restoring the
optimizer / LR-scheduler state (`base_lrs`, param-group `lr`) over the new
config. Naming only "the LR change had no effect" without the mechanism, or
blaming the Hydra override, is partial.

## Observable impact

- Primary source metric: `actor/lr` (logged by the engine from
  `lr_scheduler.get_last_lr()[0]` after each step), compared with the effective
  config printed at the start of `outputs/stage2/train.log`
  (`actor_rollout_ref.actor.optim.lr`: `2e-06`).
- Recorded run (`workspace/outputs/stage1`, `workspace/outputs/stage2`, both
  `train.log` and `metrics.jsonl`): stage 1 steps 1-2 `actor/lr` 1e-06 (as
  configured); stage 2 resumes at step 3 (`Loaded optimizer`, `Loaded
  lr_scheduler` from `global_step_2` in the log) and logs `actor/lr` 1e-06 at
  steps 3, 4 and 5.
- Recorded run reward: stage 1 0.500 / 0.484, stage 2 0.562 / 0.484 / 0.539;
  `actor/grad_norm` 0.10-0.13 after step 1; `actor/ppo_kl` |1e-5| scale throughout.
- Control reference (author-side; not shipped in the workspace and not expected
  in the diagnosis). Paired comparison from one stage-1 checkpoint (same image,
  data, seed and budget): the unchanged loader logs `actor/lr` 1e-06 at steps 3-5 (reward 0.578 /
  0.504 / 0.551); a control whose loader keeps the configured learning rates
  (logging `Checkpoint base_lrs [1e-06] differ from configured [2e-06]; using
  configured`) logs `actor/lr` 2e-06 at steps 3-5 (reward 0.563 / 0.488 / 0.543).
- The training reward does not separate the arms within three resumed steps; the
  defect is that the intended intervention (doubling the learning rate) silently
  never happened, so any conclusion drawn from stage 2 about lr 2e-6 is about
  lr 1e-6.

## Trigger conditions

- Resume from a verl FSDP checkpoint (optimizer and extra state loaded, the
  default `checkpoint.load_contents`) with a different `optim.lr` than the run
  that wrote it. Any LambdaLR-based schedule is affected (constant, cosine,
  warmup), since all scale `base_lrs`.

## Wrong attributions (must not pass)

- "The higher learning rate took effect but three steps are too few to show a
  difference": `actor/lr` is 1e-06 after the resume, so the rate never changed.
- Hydra override not applied / wrong launch command: the stage-2 effective config
  prints `'lr': 2e-06`.
- Warmup schedule holding the lr down: `lr_warmup_steps=-1` and
  `lr_warmup_steps_ratio=0.0` give a constant schedule; stage 1 logs the
  configured 1e-06 from step 1.
- Stale or wrong logging of `actor/lr`: the value comes from the scheduler that
  sets the optimizer's lr; the control with the patched loader logs 2e-06 under
  the same logging code.
- Resuming from the wrong checkpoint or not resuming: the log shows the model,
  optimizer and lr_scheduler loaded from `global_step_2` and training continues
  at step 3.
- Small `actor/ppo_kl` (about 1e-5 in both stages) read as "the policy is not
  moving, the lr is too small": that magnitude is normal for this GRPO setup
  (stage 1 at the intended 1e-6 already shows it while the reward is ~0.5 and
  `grad_norm` ~0.12); it is not evidence about the resume.
- Reward or data problems: rewards are ~0.5-0.58 in both stages and match the
  control.

## Documentation reconciliation

- `actor/lr` = 1e-06 in stage 2 is the symptom; the cause is the state restored by
  `load_checkpoint`, not the scheduler type or the logging.
- Restoring optimizer/scheduler state is correct for an exact continuation; the
  defect is that a changed `optim.lr` is accepted, printed in the config, and then
  silently discarded.
