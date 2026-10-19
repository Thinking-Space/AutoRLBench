# MoE auxiliary gradients miss token normalization

Training Qwen/Qwen1.5-MoE-A2.7B-Chat with GRPO on GSM8K exposes a loss callback that omits the token count required to normalize native MoE auxiliary gradients.

| | |
| --- | --- |
| **Author** | Gaoyuan |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `moe` `gradient-normalization` |
| **Model** | Qwen/Qwen1.5-MoE-A2.7B-Chat |
| **Dataset** | openai/gsm8k main; 7473 train rows, 64 held-out validation rows from the official 1319-row test split |
| **Framework** | verl, `scripts/train.sh` |
| **Environment** | `verl-moe-token`; solution requires 2 GPUs with at least 56 GiB free per card |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 150 steps each |

## Difficulty

`MegatronEngineWithLMHead` in `verl/workers/engine/megatron/transformer_impl.py` returns a two-item loss callback while per-token loss, CP2 and nonzero z-loss are enabled. MCore receives no token count for final normalization. Auxiliary gradients bypass manual policy-loss scaling. The two supplied workspaces have identical training configurations, launch scripts and environments; their only training-code difference is the callback repair. The source documentation also discloses an execution-time W&B finish lifecycle difference.

| Held-out accuracy (64 questions) | Defective | Repaired |
| --- | ---: | ---: |
| Step 100 | 0% | 45.3125% |
| Step 150 | 0% | 54.6875% |

Defective W&B history ends at step 148, but console history and final validation reach 150. Repaired W&B reaches 150. The histories remain separate; no missing records are synthesized. One run per arm supports the observed gap, without establishing cross-seed variance or a general collapse claim.

## Reference solution

**issue**: identify the missing native `(loss sum, token count, metrics)` contract and explain how it skips final normalization of z-loss gradients. The common router prescaling backport and padding-mask compatibility fix are already present.

**solution**: return real routed-token counts, including prompt tokens, with consistent DP/CP policy scaling. Apply an equivalent repair and complete a fresh 150-step run. The reference patch, repaired source and calibration curves are hidden in the solution oracle. The earlier native-gradient probe remains supplemental; it does not determine the full-training score.

## Verification

**issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2; passing requires at least 0.8.

**solution**: require `val-core/openai/gsm8k/acc/mean@1 >= 0.4` at step 150, accuracy <= 1.0, using only fresh `experiment_logs`. This draft threshold is calibrated on 1 run per arm. Final margins are 0.4 above defective and 0.146875 below repaired; the lowest observed repaired point clears it by 0.053125. Neither run records step-0 validation, so the threshold cannot be certified above the initial checkpoint. The grader requires the step-150 evaluation and rejects step-0-only or earlier-best logs. No variance claim is made.

The new image is **not yet built**. Offline historical grader replay and static validation are recorded separately; no fresh GPU training, online preparation, or agent/judge evaluation was run here. Native Docker uses the two visible GPUs; Harbor uses the task compose overlay. Both evaluations inherit GPU visibility from task metadata. Weights, datasets and checkpoints are omitted. The supplied material lacks W&B binaries and rollout sample dumps. Framework/new glue use Apache-2.0, GSM8K uses MIT, and the model has its own license in `environment/model-LICENSE`.
