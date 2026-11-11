# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root causes

- `runtime/checkpoint_policy.py::validate_checkpoint_schedule`
  - kind: `guard`
  - observed: rejects `save_interval >= num_rollouts` even when the run is an explicitly requested one-rollout debug smoke; it does not account for the trainer's forced final-rollout save.
  - expected: allow the debug smoke when the final rollout is intentionally the only checkpoint, while retaining the interval validation for ordinary runs.
- `runtime/mtp_training.py::train_step`
  - kind: `optional-loss interface`
  - observed: `compute_auxiliary_loss` returns `None` when `mtp_loss_scaling_factor=0.0`, but `train_step` unconditionally adds that optional value to the policy loss.
  - expected: skip the disabled auxiliary term, preserve the policy loss, and complete an optimizer step.

The two failures are coupled by the smoke protocol: the checkpoint guard prevents the one-rollout validation from reaching training, and bypassing it exposes the zero-loss interface error.

## Observable impact

- The reproduction requests one rollout and one save interval with debug-only mode enabled.
- The checkpoint guard rejects the run before training and writes no checkpoint.
- The zero auxiliary-loss path then completes zero optimizer steps and reports a `TypeError` when tested directly.
- The policy loss remains a valid finite value (`0.25`); the failure is in optional-loss handling and checkpoint scheduling, not reward quality, model capacity, or numerical instability.

## Wrong diagnoses

Reject diagnoses blaming the model, reward model, network, GPU memory, or a NaN/overflow. The failure occurs in local schedule validation and Python-side loss composition before any external service or GPU computation is required.
