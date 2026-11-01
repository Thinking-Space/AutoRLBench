# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

In `verl/workers/rollout/vllm_rollout/vllm_async_server.py`,
`engine.list_loras()` returns integer adapter IDs, but the defective code tests
membership with `VLLM_LORA_NAME`, a string. `lora_loaded` is therefore false
and no `LoRARequest` is attached to generation. vLLM samples the PiSSA residual
base while the actor computes log probabilities with the PiSSA adapter.

## Evidence

- Final `val-health/actor_rollout_policy_agreement` is
  `0.10476375009925555`; actor/rollout MAE is `2.2560474629304847`.
- Base-reference consistency is exactly `1.0`, ruling out the separate
  residual-reference defect in this task.
- All steps 0–4, rollouts and validation generations exist. Zero task reward
  and gradient are the downstream effect of poor residual-base samples, not a
  launch failure or the root-cause attribution.
- The matched integer-ID control reaches actor/rollout agreement
  `0.8907221660212565`.

Wrong attributions include simply saying the model is too weak, the questions
are too hard, reward is sparse, training did not improve, or W&B failed.
