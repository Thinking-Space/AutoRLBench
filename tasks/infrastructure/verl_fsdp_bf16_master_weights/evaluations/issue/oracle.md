# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only the
`actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` override in `run_grpo_gsm8k.sh` differs).

Expected verdict: **unhealthy**.

## Root cause

- `run_grpo_gsm8k.sh` (`actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`)
  - kind: `config`
  - observed: `verl/workers/engine/fsdp/transformer_impl.py::_build_module` loads the actor with
    `torch_dtype=bf16`, so the FSDP-sharded parameters and the AdamW moments built over them are
    bf16. bf16 resolves a weight w only to about |w|/256-|w|/128, while AdamW's step is about the
    learning rate (2e-6); for weights above about 1e-3 the step is below half the spacing and rounds
    to no change. The policy moves only through its smallest weights: `actor/kl_loss` 0.0016 (step
    10) and 0.014 (step 60) against 0.021 and 0.070 in the control.
  - expected: verl's default `model_dtype: fp32` (fp32 master weights and optimizer state, bf16
    compute through FSDP mixed precision).

Naming the override with its causal role (bf16 master weights → AdamW updates rounded away for most
weights → much slower policy movement → held-out accuracy far below the fp32 control) is a complete
attribution. Do not require additional causes.

**The dtype override and the rounding mechanism are required.** "Learning is slow", "the learning
rate is too low" or "the KL penalty is too strong" without tracing it to bf16 parameter storage earns
at most partial root-cause credit. Blaming the learning rate, the KL coefficient, the reward or
data, model capacity, AdamW epsilon, or calling the run healthy is wrong and gets no root-cause
credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` 0.434 (0) → 0.438 (10) →
  0.447 (20) → 0.460 (30) → 0.462 (40) → 0.455 (50) → 0.452 (60); re-scored from `val_generations`:
  0.4625, 0.4602, 0.4640, 0.4738, 0.4761, 0.4685, 0.4655.
- `actor/kl_loss` 0.0006 (1) → 0.0016 (10) → 0.0041 (20) → 0.0072 (30) → 0.0114 (40) → 0.0140 (60);
  entropy 0.48 → 0.24; `actor/grad_norm` 0.60-0.70 and `actor/lr` 2e-6 throughout, no NaN.
- `critic/score/mean` 0.42 (1) → 0.39 (10) → 0.43 (20-30) → 0.51 (60), rising slowly.
- The effective config (`train.log`) shows `model_dtype: bf16` for the actor's `fsdp_config`.

## Trigger conditions

- bf16 parameter storage combined with an optimizer step much smaller than the parameters' bf16
  resolution, which holds for RL fine-tuning learning rates (1e-6 to 1e-5) on pretrained weights.
- verl accepts `model_dtype=bf16` for a trainable actor without warning and reports no update-size
  or parameter-change metric.

## Documentation reconciliation

- The small KL, the slow entropy decline and the flat held-out accuracy are symptoms of the same
  rounding; gradients, losses and the learning rate are normal by construction.
