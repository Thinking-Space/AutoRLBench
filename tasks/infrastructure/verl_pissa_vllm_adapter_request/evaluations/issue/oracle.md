# Canonical issue oracle

Reconciled from the task's frozen evidence: `observed_run/` (`train.log`,
`metrics/all_steps.jsonl`, `debug/*.rank0.json`, `rollouts/*.jsonl`,
`validation_generations/*.jsonl`) and the verl fork shipped in the workspace.
The repaired-run figures are author-side reference and live in Documentation
reconciliation below.

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`, the generate
  request path, gates the LoRA with
  `lora_loaded = VLLM_LORA_NAME in loaded_loras`, where
  `loaded_loras = await self.engine.list_loras()`.
  - kind: `implementation_bug`
  - observed: `verl/workers/rollout/vllm_rollout/utils.py` defines
    `VLLM_LORA_INT_ID = 123` and `VLLM_LORA_NAME = "123"`, but vLLM's
    `list_loras()` returns the registered adapters' *integer* IDs (a
    `set[int]` — the same contract in the frozen run's `vllm==0.11.0` and in
    the image's vLLM). The membership test `"123" in {123, ...}` is therefore
    always False, `lora_request` stays `None`, and every vLLM generation
    request is sent without the LoRA. vLLM samples the PiSSA residual base
    while the actor computes log probabilities with the PiSSA adapter
    active: every rollout is off-policy, all rewards are zero, and the run's
    own final health check collapses.
  - expected: test membership with `VLLM_LORA_INT_ID` — the integer the
    engine actually returns and that `vllm_rollout.py::update_weights`
    registered the adapter under via
    `TensorLoRARequest(lora_int_id=VLLM_LORA_INT_ID, ...)` — and attach the
    `LoRARequest` to each generate call.

Naming this cause with its correct causal role (the string-vs-integer adapter
ID mismatch in the request path → generation runs adapter-less on the residual
base while the actor trains with the adapter → degenerate rollouts, zero
reward/advantage/gradient, actor/rollout agreement 0.10) is a complete
attribution. Do not require additional causes.

**Identifying the mismatched check is required.** "The rollout diverges",
"agreement is low", "the outputs are garbage" without the string-vs-integer
adapter-ID mechanism are symptoms, worth at most partial root-cause credit.
Blaming the model's capacity, question difficulty, reward sparsity, rollout
weight synchronization, the base-reference machinery, or a wandb/network
crash is wrong and gets no root-cause credit.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`observed_run/train.log`, `observed_run/metrics/all_steps.jsonl`,
`observed_run/debug/*.rank0.json`, the rollout and validation dumps, the
launch script and the verl fork). The repaired control-run figures are
author-side reference and live in Documentation reconciliation below. Each
bullet is one item.

- The run executes its full designed budget without crashing:
  `total_training_steps: 4`, four real GRPO update steps logged (plus the
  step-0 validation); the trailing `ConnectionResetError` in `train.log` sits
  in wandb's offline service publish path after the last step — shutdown
  noise, not the failure.
- Final health check (`metrics/all_steps.jsonl`, step-4 row):
  `val-health/actor_rollout_policy_agreement` is `0.10476375009925555` with
  `val-health/actor_rollout_policy_mae` `2.2560474629304847` — the actor
  recompute and the vLLM rollout policy disagree massively.
- `val-health/base_reference_consistency` is exactly `1.0` with
  `val-health/base_reference_mae` `0.0` — the base-reference machinery is
  exact, ruling out the sibling PiSSA residual-reference defect.
- Training is flat zero everywhere: `actor/kl_loss`, `actor/base_model_kl`,
  `critic/rewards/mean` and `actor/grad_norm` are `0.0` at every update, and
  held-out accuracy is `0.0` at steps 0-4 — zero reward makes every GRPO
  group advantage zero, so the policy never moves and never learns.
- `observed_run/debug/unknown_exp.rollout_sync_initial.rank0.json`:
  `is_svd_init: true`, `peft_type: "none"`,
  `rollout_sync_mode: "svd_init_original_pretrained_full_weights"`;
  `unknown_exp.rollout_sync_latest.rank0.json`: `peft_type: "PeftType.LORA"`,
  `rollout_has_peft_config: true`, `rollout_sync_mode: "lora_delta"` — the
  weight-sync pipeline into vLLM ran its full designed sequence (`train.log`
  logs each probe write under `.../recorded_run/rollout_policy_debug/`).
- `observed_run/debug/unknown_exp.actor_compute_log_prob.rank0.json` and
  `unknown_exp.actor_update_policy.rank0.json`: `has_peft_config: true`,
  `active_adapters: ["default"]` — the actor evaluates WITH the PiSSA
  adapter; `unknown_exp.pre_update.rank0.json` and
  `unknown_exp.actor_update.rank0.json` show `diff_abs_mean 0.0` and
  `masked_base_kl 0.0` with `base_ref_mean` equal to `old_mean` (`-6.2511`
  and `-5.3433`) — the actor side is internally exact.
- The code site: `vllm_async_server.py`'s generate path computes
  `lora_loaded = VLLM_LORA_NAME in loaded_loras` from
  `await self.engine.list_loras()`; `utils.py` defines `VLLM_LORA_INT_ID = 123`
  and `VLLM_LORA_NAME = "123"`; vLLM's `list_loras()` returns integer IDs, so
  the string membership test is always False and `lora_request` stays `None`
  for every request.
- The load side is type-correct: `vllm_rollout.py::update_weights` registers
  the adapter with `remove_lora(VLLM_LORA_INT_ID)` /
  `TensorLoRARequest(lora_int_id=VLLM_LORA_INT_ID, ...)` — the adapter is in
  the engine under integer ID 123; only the request-side check mismatches the
  type.
- The `train.log` init sequence shows both sides claiming LoRA: "Applying
  LoRA to actor module" → "Loading pre-trained LoRA adapter to actor from:
  .../pissa_adapter" → "[SVD-init ref] Saved 96 initial adapter parameters
  for correct ref computation." → "[SVD-init ref] Loaded frozen reference
  adapter '_base_ref' for correct base-model KL computation.", while the
  vLLM server starts with `--enable_lora` (dynamic-loading warnings); the
  resolved config confirms `lora_rank 8`, `lora_alpha 8`,
  `lora_residual_base: True`, `rollout.name vllm`, `rollout.mode async`.
- Rollout and validation outputs are degenerate: `rollouts/{1,2,4}.jsonl`
  and `validation_generations/{0,2,4}.jsonl` contain non-answers like
  "Robertated the payualz 28 w 61 source)" and
  "Rid is 40 1200000000000000000 120 1", all with score/acc `0.0` — sampling
  from the adapter-less residual base.

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
