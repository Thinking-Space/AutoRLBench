# Zero MTP loss path blocks training smoke and checkpointing

During GRPO training of Qwen3.5-9B on a mixed code-repair corpus, a zero-scaled auxiliary MTP loss is consumed as if present and the checkpoint guard rejects the one-rollout smoke needed to expose it.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Algorithm |
| **Domain** | Code RL |
| **Tags** | `mtp` `zero-loss` `checkpoint-guard` `rl-training` |
| **Framework** | miles (GRPO), Megatron training, SGLang rollout |
| **Run time** | minutes (one-rollout smoke, stopped before a useful update) |
| **Resources** | 16 × H100 (80 GB) |

## Difficulty

The training configuration intentionally sets the auxiliary MTP loss scale to
`0.0` for this RL path. The loss producer therefore returns no auxiliary term,
but the training step still adds it to the finite policy loss. The direct
reproduction requests one rollout with `save_interval=1`; the checkpoint guard
rejects that schedule before training because it does not recognize the
debug-only final checkpoint.

The shipped smoke reports `optimizer_steps=0`, no checkpoint, and the concrete
`TypeError` from the optional-loss interface. These are local schedule and loss
composition failures, not evidence of a bad reward model, network failure,
GPU memory failure, NaNs, or insufficient model capability.

The historical GPU smoke established seven normal optimizer updates after the
MTP hook was migrated, and five focused CPU tests passed, but there is no clean
same-seed defective-versus-fixed training comparison for a solution threshold.
Issue-only: the package preserves the diagnosis and a one-step reproduction.

## Reference solution

- **issue**: identify both coupled defects: `validate_checkpoint_schedule` rejects a valid one-rollout debug smoke when the final rollout is intentionally the only save, and `train_step` unconditionally consumes the `None` returned for `mtp_loss_scaling_factor=0.0`; reject GPU, network, reward, and numerical-instability diagnoses.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
