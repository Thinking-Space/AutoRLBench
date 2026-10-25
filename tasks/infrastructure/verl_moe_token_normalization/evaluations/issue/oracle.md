# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `workspace/src/framework/verl/workers/engine/megatron/transformer_impl.py`,
  `MegatronEngineWithLMHead` loss postprocessing callback:
  - kind: symbol
  - observed: returns a two-item loss/metrics tuple under calculate_per_token_loss=True.
    MCore receives no token count; schedule total_num_tokens remains zero, so final
    token normalization is skipped for native auxiliary gradients.
  - expected: supply the native (loss sum, token count, metrics) contract and scale
    the already-normalized policy loss consistently with DP/CP and microbatches.
    Count real routed tokens, including prompts, rather than response-only loss_mask.

## Trigger conditions

The recorded real_gsm8k.yaml enables calculate_per_token_loss=True, CP2, EP2,
TP1, and nonzero moe_z_loss_coeff=0.001. The router pre-scales the z-loss
gradient by num_tokens through MoEAuxLossAutoScaler, expecting MCore to divide
it back by the token count; the auxiliary autograd gradients bypass the manual
scaling applied to the main policy loss (the `has_auxiliary_loss` branch skips
the DP*CP division). The router prescaling backport and padding-mask
compatibility fixes are already present; they are not the missing fix.

## Observable impact

- actor/grad_norm is 223.22 at W&B step 1 and 144.49 at step 148, far above
  clip_grad=1.0 from the first step onward.
- critic/rewards/mean falls from 0.25 (step 1) to 0 (step 148);
  response_length/mean rises from 233 to 1024 and saturates the response limit.
- Held-out val-core/openai/gsm8k/acc/mean@1 in stdout.log is 0.0 at step 100
  and 0.0 at step 150.

## Wrong diagnoses

- The two missing W&B tail records (steps 149-150) are a logging limitation;
  training and final validation reach step 150 in stdout.log.
- A missing router prescaling backport or padding-mask fix: both are present.
- Synthetic data or an absent full run.
- Low training reward alone, without connecting it to the gradient scale.
- Disabling z-loss or per-token mode avoids the trigger but does not repair the
  normalization contract; naming it as the root cause is wrong.
