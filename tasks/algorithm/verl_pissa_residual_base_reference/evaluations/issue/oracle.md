# Canonical issue oracle

Reconciled from the task's frozen evidence: `observed_run/` (`train.log`,
`metrics/all_steps.jsonl`, `debug/*.rank0.json`) and the verl fork shipped in the
workspace. The repaired-run figures are author-side reference and live in
Documentation reconciliation below.

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/fsdp_workers.py::_compute_base_log_prob_from_actor`, the
  `_is_lora and _is_svd_init` branch, computes every "base-model" reference
  log-prob the same way the standard LoRA branch right below it does:
  `meta_info["is_lora"] = True` → `compute_log_prob`, whose forward is wrapped
  in `actor_module.disable_adapter()` (comment: "Standard LoRA/plus/b0:
  disable_adapter() exposes the underlying base model").
  - kind: `implementation_bug`
  - observed: this run trains in PiSSA mode (`lora_residual_base: True`, rank
    8, alpha 8), so the plain base checkpoint holds
    `W_residual = W_pretrained - A_init B_init`, not the pretrained weights.
    Disabling the adapter therefore evaluates `W_residual`, and every consumer
    of the function — GRPO's `compute_ref_log_prob` (`ref_log_prob`, the
    `actor/kl_loss` regularization with `kl_coef 0.05`), the `update_policy`
    base-KL logging (`base_ref_log_prob` → `actor/base_model_kl`), and the
    "actual" side of the final health check — is measured against the residual
    base instead of the pretrained policy. The fork itself loads a frozen
    `_base_ref` adapter (the initial `A_init`, `B_init`) and ships
    `_svd_init_adapter_ctx()` to evaluate it on top of `W_residual`, which
    reconstructs `W_pretrained` exactly — but only
    `compute_expected_svd_base_ref_log_prob` (the validation-health probe)
    uses that context; the training path never does.
  - expected: for SVD-init (PiSSA) LoRA, base-reference log-probs must be
    computed with the frozen initial adapter active on top of the residual
    base — the machinery this fork already ships and loads at startup — so the
    KL reference equals the pretrained policy.

Naming this cause with its correct causal role (PiSSA rewrites the base
checkpoint to the residual → the adapter-disabled actor is `W_residual`, not
the pretrained policy → the base-KL reference used by GRPO training is the
wrong distribution) is a complete attribution. Do not require additional causes.

**Identifying the wrong reference is required.** "KL is large", "the model does
not learn", "the run is unstable" without the residual-base/adapter-disabled
mechanism are symptoms, worth at most partial root-cause credit. Blaming the
KL coefficient, the learning rate, the tiny dataset, the validation accuracy,
rollout synchronization, or a wandb/network crash is wrong and gets no
root-cause credit.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`observed_run/train.log`, `observed_run/metrics/all_steps.jsonl`,
`observed_run/debug/*.rank0.json`, the launch script and the verl fork). The
repaired control-run figures are author-side reference and live in
Documentation reconciliation below. Each bullet is one item.

- The run executes its full designed budget without crashing:
  `total_training_steps: 4`, four real GRPO update steps logged (plus the
  step-0 validation), final `critic/rewards/mean` `0.125` and
  `actor/grad_norm` `2.3105337619781494`; the trailing
  `ConnectionResetError` in `train.log` sits in wandb's offline service publish
  path after the last step — shutdown noise, not the failure.
- `actor/kl_loss` (the KL-to-reference term, `kl_coef 0.05`) is
  `0.7618025616` at step 1 and stays `0.76`-`0.98` across all four updates;
  with a correct pretrained reference the policy is one tiny update
  (lr `2e-05`) away from init at step 1, so this term should start near zero.
- `actor/base_model_kl` is large-negative throughout: `-4.02`, `-4.61`,
  `-4.93`, `-4.61` for steps 1-4, with `actor/base_model_kl_pre_update`
  `-4.08` to `-4.65` — the logged "base" sits several nats below the policy
  everywhere, while against the true pretrained reference this quantity is
  near zero.
- The per-step rollout-correlation metrics show the actor recompute and the
  rollout policy stay in agreement (e.g. step 4
  `training/rollout_actor_probs_pearson_corr` `0.9903733134269714`,
  `training/rollout_probs_diff_mean` `0.03785998374223709`,
  `rollout_corr/rollout_ppl` `3.1995` vs `rollout_corr/training_ppl` `3.1945`,
  `rollout_corr/chi2_seq` `0.48`): rollout synchronization is not the failure
  here.
- The fork ships the correct-reference machinery and even exercises it at
  every validation — `compute_expected_svd_base_ref_log_prob`
  (`verl/workers/fsdp_workers.py`, invoked from `ray_trainer.py`'s validation
  path) evaluates the frozen `_base_ref` adapter on top of `W_residual`
  (i.e. `W_pretrained`) via `_svd_init_adapter_ctx()` — yet the
  training-path reference still comes from the adapter-disabled actor; the
  code contradicts itself about which base the reference should use, and the
  per-step KL symptoms are exactly what that contradiction predicts.
- `observed_run/debug/unknown_exp.rollout_sync_initial.rank0.json`: `is_svd_init: true`,
  `peft_type: "none"`,
  `rollout_model_path: ".../qwen2.5-0.5b-pissa-r8/residual_base"`,
  `lora_adapter_path: ".../qwen2.5-0.5b-pissa-r8/pissa_adapter"`,
  `rollout_sync_mode: "svd_init_original_pretrained_full_weights"` — the base
  checkpoint in play is the PiSSA residual base.
- `observed_run/debug/unknown_exp.pre_update.rank0.json`: `masked_base_kl` `-4.075873`
  with `base_ref_mean` `-16.1617` against `old_mean` `-18.4523`
  (`diff_abs_mean` `3.4609`) — the pre-update "reference" log-probs are ~3.5
  nats/token away from the policy's own.
- `observed_run/debug/unknown_exp.actor_compute_log_prob.rank0.json`: `is_lora_flag: true`,
  `active_adapters: ["default"]`, `disable_adapter_layers: false` — the
  base-ref forward really runs through the adapter-disabled actor.
- The code site: in `_compute_base_log_prob_from_actor` the `_is_svd_init`
  branch is equivalent to the standard LoRA branch under it (both set
  `meta_info["is_lora"] = True` and call `compute_log_prob`, which wraps the
  forward in `actor_module.disable_adapter()`), while the correct
  `_svd_init_adapter_ctx()` is entered only by
  `compute_expected_svd_base_ref_log_prob`.
- The `train.log` init sequence proves the correct reference is loaded and
  then bypassed: "Loading pre-trained LoRA adapter to actor from:
  .../pissa_adapter" → "[SVD-init ref] Saved 96 initial adapter parameters
  for correct ref computation." → "[SVD-init ref] Loaded frozen reference
  adapter '_base_ref' for correct base-model KL computation."; the resolved
  config confirms `lora_residual_base: True`, `lora_rank 8`, `lora_alpha 8`.
- Held-out accuracy on the 8-sample validation subset bounces `0.0` → `0.125`
  → `0.0` → `0.125` → `0.0` (steps 0-4) and training reward is `0` for the
  first three updates (`0.125` at the last) — no learning signal anywhere.

## Trigger conditions

- Porting PiSSA / SVD-init LoRA into an RL framework whose reference-policy
  code assumes "disable_adapter() exposes the base model". The assumption
  holds for plain LoRA (the base checkpoint still holds the pretrained
  weights) and silently breaks once PiSSA rewrites the base checkpoint to
  `W_residual = W_pretrained - A_init B_init`: the adapter-disabled model is
  then a far-away distribution, and every KL/reference computation keyed to it
  (GRPO `kl_loss`, base-model KL logging, health checks) is wrong while the
  run keeps executing normally.

## Documentation reconciliation

- Control reference (author-side, not scored): the matched repaired run —
  changing only the reference construction to the frozen initial adapter on
  top of `W_residual` — reaches base-reference consistency `1.0`. These
  figures are rebuttal context for wrong diagnoses, not evidence items and
  not part of the evidence denominator.
- "The run is too short to judge" is wrong: `total_training_steps: 4` is the
  designed budget of this calibration recipe (16 train / 8 validation
  samples); the failure is the wrong reference (see the KL items above), not
  the horizon.
- "The low validation accuracy is the disease" is wrong: on 8 samples the
  accuracy bounces between `0` and `0.125` in any reading; it is a symptom
  downstream of training against the wrong reference.
- "The KL coefficient is too high" is wrong: `kl_coef 0.05` is a standard
  value; the KL term is large because its reference is the wrong
  distribution, not because of the coefficient.
- "Rollout desynchronization" is wrong: the per-step
  `rollout_actor_probs_pearson_corr` stays at `0.99` and the sync dump shows
  the intended `svd_init_original_pretrained_full_weights` rollout mode.
- "A crash killed the run" is wrong: the single traceback at the end of
  `train.log` is wandb's offline service connection during shutdown, after
  the last step; there are no errors, NaNs or aborts during training.
