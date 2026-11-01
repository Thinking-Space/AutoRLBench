# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

PiSSA initialization stores `W_residual = W_pretrained - A_init B_init` in the
plain base checkpoint. The PiSSA branch of
`verl/workers/fsdp_workers.py::_compute_base_log_prob_from_actor` incorrectly
uses the adapter-disabled actor as the base reference. That produces log
probabilities from `W_residual`, not the original pretrained policy required by
the base-KL calculation.

The correct path temporarily evaluates the frozen initial PiSSA adapter (the
initial `A_init`, `B_init` tensors) on top of `W_residual`, reconstructing the
source policy without changing the trained adapter.

## Evidence

- The run completes four real GRPO updates with reward `0.125` and final
  gradient norm `2.3105337619781494`; this is not a launch failure.
- Actor/vLLM policy agreement remains healthy at `0.8935824763276098`, ruling
  out rollout synchronization as the cause in this task.
- Final `val-health/base_reference_consistency` is only
  `0.045778327704709` and `base_reference_mae` is `3.083944494074041`.
- The matched repaired run reaches base-reference consistency `1.0` while
  changing only this reference construction.

Wrong attributions include the low validation accuracy, the short run, KL
coefficient, W&B shutdown warnings, or a generic claim that training failed.
