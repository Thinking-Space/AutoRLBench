# Canonical issue oracle

Source: verl-project/verl issue 6609 and the supplied reference.patch, sealed in reference/.
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
TP1, and nonzero moe_z_loss_coeff=0.001. Auxiliary autograd gradients bypass
manual scaling of the main policy loss. The shared router prescaling backport
and padding-mask compatibility fixes are already present; they are not the missing fix.

## Observable impact

The supplied W&B history has 148 records; the console history reaches 150.
At W&B steps 1 and 148 actor/grad_norm is 223.222475206911 and
144.49191694520493, critic/rewards/mean is 0.25 and 0, and
response_length/mean is 233 and 1024. These are observations from one run,
not proof that this defect alone caused collapse. Held-out validation is in stdout.log.

## Wrong diagnoses and evidence limits

Do not attribute the defect to the two missing W&B tail records, synthetic data,
an absent full run, or a missing router backport. Training reaches step 150;
missing W&B steps 149-150 are a logging limitation. Low training reward alone
does not establish held-out health or causation. Disabling z-loss or per-token
mode avoids the trigger but does not repair the normalization contract.
No calibrated repair threshold or agent evaluation is included in this task.
