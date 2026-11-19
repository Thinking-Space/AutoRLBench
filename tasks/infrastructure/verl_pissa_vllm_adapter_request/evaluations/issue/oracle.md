# Canonical issue oracle

Reconciled from the task's frozen evidence: `observed_run/` (`train.log`,
`metrics/all_steps.jsonl`, `debug/*.rank0.json`, `rollouts/*.jsonl`,
`validation_generations/*.jsonl`) and the verl fork shipped in the workspace.
The repaired-run figures are author-side reference and live in Documentation
reconciliation below.

Expected verdict: **unhealthy**.

## Root cause

The vLLM request path has a string/integer adapter-ID contract mismatch. `vllm_async_server.py` computes `lora_loaded = VLLM_LORA_NAME in loaded_loras`, while `engine.list_loras()` returns the registered adapter IDs as integers. `utils.py` defines the same ID as integer `VLLM_LORA_INT_ID = 123` and string `VLLM_LORA_NAME = "123"`; the adapter is registered under the integer ID, but the request membership test uses the string, so it is always false. `lora_request` stays `None`, and every generation runs without the loaded PiSSA/LoRA adapter. The actor still evaluates with the adapter, so vLLM and actor use different policies. This is the essential cause; residual-base sync details and zero rewards are downstream context.

## Observable impact

Every item below is derivable from the shipped workspace. Equivalent observations are accepted; exact values and all bullets are corroboration, not separate hard requirements.

1. **Full-run/health context:** the four-step run completes its designed budget without a training crash; the trailing wandb `ConnectionResetError` is shutdown noise.
2. **Policy divergence:** actor/rollout agreement is about 0.105 with large MAE/log-prob divergence, while actor-side adapter/reference checks are internally consistent (`base_reference_consistency=1.0`); the sync probes show the adapter is loaded on the actor/engine side but later generation requests omit it.
3. **Training impact:** rewards, group advantages, actor loss and gradients are zero or nearly zero, held-out accuracy is zero, and rollouts/validation become malformed or degenerate after the adapter omission.

The exact request-side type mismatch and its effect on `lora_request` are stronger evidence than any single reward or agreement number.

## Trigger conditions

- Porting PiSSA / LoRA RL into a framework whose async vLLM request path
  re-checks adapter liveness by name while the engine's `list_loras()` API
  returns integer IDs — an implicit type-contract violation between verl's
  request gate and vLLM's adapter registry. Both sides independently look
  correct (the adapter loads into the engine; the actor trains with it), the
  run completes its full budget with no error, and the breakage surfaces only
  through off-policy samples: degenerate generations, zero reward, and the
  actor/rollout agreement health check — the classic training–inference
  inconsistency class.

## Documentation reconciliation

- Control reference (author-side, not scored): the matched repaired run —
  changing only the membership test to the integer ID — reaches actor/rollout
  agreement `0.8907221660212565`. These figures are rebuttal context for
  wrong diagnoses, not evidence items and not part of the evidence
  denominator.
- "The model is too weak / the questions are too hard / reward is sparse" is
  wrong: the same 0.5B model answers in the repaired control; the zero reward
  is downstream of degenerate rollouts, not a property of the task.
- "Training did not improve" is a symptom, not a cause: with zero group
  advantage the policy has no gradient to learn from.
- "Rollout desynchronization" is wrong: the sync probes ran the full designed
  sequence (`svd_init_original_pretrained_full_weights` → `lora_delta` with
  the PEFT config present) and the actor-side probes are internally exact.
- "The PiSSA base-reference defect" is wrong: `base_reference_consistency` is
  exactly `1.0` in this run.
- "A crash killed the run / W&B failed" is wrong: the single traceback at the
  end of `train.log` is wandb's offline service during shutdown, after the
  last step; there are no errors, NaNs or aborts during training.
- "The run is too short to judge" is wrong: `total_training_steps: 4` is the
  designed budget of this calibration recipe (16 train / 8 validation
  samples); the failure is quantified by agreement `0.1048`, not the horizon.
