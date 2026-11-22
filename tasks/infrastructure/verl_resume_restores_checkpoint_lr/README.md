# Resuming from a checkpoint silently restores the old learning rate

Training Qwen/Qwen3-4B with verl GRPO on an integer-answer math pool in two stages: stage 1 trains at lr 1e-6 and checkpoints, stage 2 resumes from that checkpoint with `actor_rollout_ref.actor.optim.lr=2e-6`. The resumed run prints the new value in its effective config but keeps training at 1e-6, because loading the checkpoint restores the optimizer's and LR scheduler's learning rates over the configured one.

| | |
| --- | --- |
| **Author** | Sikuan Yan |
| **Affiliation** | LMU Munich |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `checkpoint-resume` `lr-scheduler` `silent-config-override` |
| **Framework** | verl v0.8.0 tag (commit `7aed6b2`, shipped in the workspace; its version file reads 0.8.0.dev), GRPO, FSDP engine; inspect_ai 0.3.150 installed by `prepare.sh` |
| **Run time** | ~36 minutes (2 + 3 steps; stage 1 writes a ~47 GB checkpoint) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

`run_resume.sh` runs `run_grpo.sh` twice. Stage 1 (`outputs/stage1`) trains steps 1-2 at `LR1=1e-6` and saves `checkpoints/global_step_2`; stage 2 (`outputs/stage2`) resumes from it with `trainer.resume_mode=resume_path` and `LR2=2e-6`. The stage-2 effective config shows `'lr': 2e-06`, the log shows the model, optimizer and `lr_scheduler` loaded from `global_step_2`, and training continues at step 3 with healthy rewards. Only the logged `actor/lr` gives it away: 1e-06 at steps 3, 4 and 5.

The engine (`verl/workers/engine/fsdp/transformer_impl.py`) builds AdamW and a constant-with-warmup `LambdaLR` from the new config, and then `FSDPCheckpointManager.load_checkpoint` (`verl/utils/checkpoint/fsdp_checkpoint_manager.py`) calls `optimizer.load_state_dict` (param-group `lr` and `initial_lr` back to 1e-6) and `lr_scheduler.load_state_dict` (`base_lrs = [1e-06]`). Each later `lr_scheduler.step()` recomputes `lr = base_lr × 1.0`, so the configured value is used only before the checkpoint is loaded, and nothing warns. Restoring the state is right for an exact continuation; the defect is that a changed `optim.lr` is accepted, printed, and silently discarded. Any LambdaLR schedule (constant, cosine, warmup) is affected the same way.

The run looks healthy, so the obvious readings are wrong: "the higher lr took effect but three steps are too few to show it" (the rate never changed), "the Hydra override was not applied" (the config prints 2e-06), "warmup is holding the lr down" (`lr_warmup_steps=-1`, ratio 0.0, and stage 1 logs the configured 1e-06 from step 1), "`actor/lr` logging is stale" (the same logging shows 2e-06 when the loader keeps the configured rate), and "the tiny `actor/ppo_kl` (~1e-5) means the lr is too small" (that magnitude is normal here: stage 1 shows it at the intended rate with reward ~0.5).

| step | stage | configured lr | recorded `actor/lr` | control `actor/lr` | recorded `critic/score/mean` | recorded `actor/grad_norm` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1e-6 | 1e-06 | 1e-06 | 0.500 | 0.343 |
| 2 | 1 | 1e-6 | 1e-06 | 1e-06 | 0.484 | 0.132 |
| 3 | 2 (resumed) | 2e-6 | **1e-06** | 2e-06 | 0.562 | 0.130 |
| 4 | 2 | 2e-6 | **1e-06** | 2e-06 | 0.484 | 0.103 |
| 5 | 2 | 2e-6 | **1e-06** | 2e-06 | 0.539 | 0.115 |

Horizon: 5 GRPO steps (2 + 3), no in-run validation (`trainer.test_freq=-1`; a held-out evaluation could not tell lr 1e-6 from 2e-6 within three steps anyway, so `actor/lr` against the effective config is the signal); the evidence is the effective configs and per-step metrics in `workspace/outputs/stage{1,2}/train.log` and `metrics.jsonl`, three rollout dumps, and the control. The W&B `run-*.wandb` history files were removed because they embed host paths and user/host names; `metrics.jsonl` carries the same per-step history. The recorded run ran once (data seed 1) in the shared `verl-vllm0.17.0` image on 2 × H100 with `run_resume.sh`, start model `Qwen/Qwen3-4B` (revision `1cfa9a7`), 32 prompts × 8 samples, response cap 8192, 5.5-7.3 minutes per step. The control column comes from a separate pair that resumed one stage-1 checkpoint twice under the same config: unchanged loader (`actor/lr` 1e-06 at steps 3-5, reward 0.578 / 0.504 / 0.551) and a loader that re-applies the configured learning rates after loading (`actor/lr` 2e-06 at steps 3-5, reward 0.563 / 0.488 / 0.543). The defect was first met in an automated RL experiment: after misreading a small `ppo_kl` as "lr too small", the run was resumed with a doubled lr, and the scheduler state in the checkpoint (`base_lrs: [1e-06]`) showed the change would not apply.

Data: `data/pool_clean.jsonl` (81,882 problems) is the same post-decontamination pool as `verl_pandas_float_ground_truth`: integer-answer problems from Skywork-OR1 (math split), POLARIS-53K and DAPO-Math-17k, with integer-normalized ground truths in `scripts/make_rl_data.py` and `scripts/math_reward.py`, so rewards are valid here. The pool is shipped as built and is byte-identical to the one in `verl_pandas_float_ground_truth`; its assembly and decontamination script is not included, and the decontamination does not affect this defect. Licences: POLARIS-53K and DAPO-Math-17k are Apache-2.0 on Hugging Face; the Skywork-OR1-RL-Data card states no licence, and its GitHub repository (SkyworkAI/Skywork-OR1) is Apache-2.0.

Issue-only: the reward does not separate the arms within three resumed steps, so the only gradeable signal is `actor/lr` itself; several repairs are valid (fix the loader, edit the checkpoint's scheduler/optimizer state, or start a fresh run), and a rerun would have to repeat both stages and write a ~47 GB checkpoint.

## Reference solution

- **issue**: identify that stage 2 trains at 1e-6 although `optim.lr=2e-6` is configured (`actor/lr` 1e-06 at steps 3-5 against `'lr': 2e-06` in the effective config) and attribute it to `FSDPCheckpointManager.load_checkpoint` restoring the optimizer param-group `lr`/`initial_lr` and the LR scheduler's `base_lrs` from the checkpoint over the newly configured value. A repair keeps the configured learning rates after loading (or warns/refuses on a mismatch), or edits the checkpoint state before resuming. Attributions to too few steps, a missing Hydra override, warmup, stale logging, the wrong checkpoint, or "ppo_kl is small so the lr is too small" do not pass; saying only that the lr change had no effect without the resume mechanism is partial.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. As implemented in the shared `judge.py`, passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
