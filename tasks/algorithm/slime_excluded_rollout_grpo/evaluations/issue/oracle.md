# Canonical issue oracle

Expected verdict: **unhealthy task-advantage computation**. The evidence is a CPU post-rollout replay. It establishes a wrong learning signal and does not establish a downstream score change.

## Root cause

Historical OpenClaw aborts could supply a terminal rollout with reward zero, zero loss mask, `remove_sample=True` and status `ABORTED`. The historical cost-budget callback grouped terminal rollouts but forwarded every prompt-group record to `task_advantages`. The replay isolates this call sequence in `workspace/replay.py::post_process_rewards`. `workspace/reward_common.py::collect_rollouts` groups fragments by `(group_index, rollout_id)`, and `task_advantages` mean-centers their terminal rewards and divides by sample standard deviation plus `1e-6` for the recorded GRPO setting. Those helpers do not infer eligibility.

In the clean batch, eligible rewards `1.0` and `0.6` have mean about `0.8`, giving the latter advantage `−0.707104`. The excluded zero changes the input population to `[1.0, 0.6, 0.0]` and the mean to about `0.5333`. The valid 0.6-rollout now receives `+0.132453`. Its peer's advantage also changes. The excluded rollout has effective advantage zero, so the error comes from its effect on the group statistics before masking. Either exclusion flag is sufficient for the observed failure.

The eligibility rule acts on terminal rollout records before computing task statistics. In the paired control it removes records with either exclusion flag, assigns excluded fragments zero advantage and expands each remaining rollout's value back to its fragments. A real completed rollout with task reward zero is still eligible. Singleton groups yield zero when centered. A fully excluded group yields zero for every fragment without attempting to normalize an empty set.

## Evidence and plausible incorrect diagnoses

- `workspace/batches.json` labels the synthetic clean, interrupted, real-zero, singleton and all-excluded inputs. `workspace/evidence/recorded-replay.json` shows both single-flag cases giving the same `+0.132453` as the completed zero, though their own effective advantages differ.
- `remove_sample` and `ABORTED` entries have nonzero intermediate task advantages in the defective arm and zero effective advantages. Late loss masking is insufficient because the unaffected peer set has already received distorted values.
- Both fragments of the highest-reward rollout receive the same value. The grouping operation already prevents a multi-fragment rollout from contributing its reward twice. A diagnosis of raw fragment counting alone misses the bug.
- Standard deviation convention, optimizer parameters and reward clipping do not explain the sign flip. It comes from changing the normalization population. Blanket deletion of zero-valued rewards would remove valid completed samples.
- A passing diagnosis should connect the pre-normalization eligibility omission to the sign flip using source and replay evidence. Mentioning an invalid row or late masking without explaining how that row changes valid peers earns partial root-cause credit.
