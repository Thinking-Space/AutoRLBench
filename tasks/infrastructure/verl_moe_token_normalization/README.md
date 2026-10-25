# MoE auxiliary gradients miss token normalization

Training Qwen/Qwen1.5-MoE-A2.7B-Chat with GRPO on GSM8K. The per-token loss
callback omits the token count that MCore needs to normalize the native MoE
auxiliary losses, so the z-loss gradients are never divided back.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `moe` `gradient-normalization` |
| **Model** | Qwen/Qwen1.5-MoE-A2.7B-Chat |
| **Dataset** | openai/gsm8k main; 7473 train rows, 64 held-out validation rows from the official 1319-row test split |
| **Framework** | verl, Megatron engine, `scripts/train.sh` |
| **Environment** | `verl-moe-token` (not yet built) · 2 GPUs with at least 56 GiB free per card |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 150 steps each |

## Difficulty

`MegatronEngineWithLMHead` in
`verl/workers/engine/megatron/transformer_impl.py` returns a two-item
`(loss, metrics)` callback while per-token loss, CP2 and a nonzero
`moe_z_loss_coeff` are enabled, so MCore receives no token count for final
normalization. The router z-loss gradients (pre-scaled by `num_tokens`) are
never divided back: `actor/grad_norm` is ~223 at step 1 against ~3.3 for the
repaired callback. The common router pre-scaling backport and padding-mask
compatibility fix are already present, which rules out the two nearby
explanations. The two supplied workspaces have identical training
configurations, launch scripts and environments; their only training-code
difference is the callback repair.

| Held-out accuracy (64 questions) | Defective | Repaired |
| --- | ---: | ---: |
| Step 100 | 0% | 45.3125% |
| Step 150 | 0% | 54.6875% |

Defective W&B history ends at step 148, but console history and final
validation reach 150; repaired W&B reaches 150. The histories remain
separate, no missing records are synthesized, and one run per arm supports
the observed gap without establishing cross-seed variance.

## Reference solution

- **issue**: identify the missing native `(loss sum, token count, metrics)`
  contract and explain how it skips final normalization of z-loss gradients.
- **solution**: return real routed-token counts, including prompt tokens, with
  consistent DP/CP policy scaling, then complete a fresh 150-step run. The
  reference patch, repaired source and calibration curves are hidden in the
  solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: the agent repairs the workspace and submits `report.md` without
  training; a missing report scores 0. The harness then reruns
  `scripts/train.sh` per `run_spec.json` and requires
  `val-core/openai/gsm8k/acc/mean@1` at step 150 ≥ 0.4 (cap 1.0). Only the
  step-150 evaluation of a fresh run counts: the grader reads only
  `experiment_logs`, rejects step-0-only or earlier-best logs, and extending
  the run or raising eval frequency cannot lift an unrepaired score. The
  threshold sits 0.4 above the defective run and 0.146875 below the repaired
  one; the lowest observed repaired point clears it by 0.053125. It is
  calibrated on one run per arm, so treat scores near 0.4 as untested
  territory.
