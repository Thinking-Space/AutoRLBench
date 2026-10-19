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
| **Environment** | `verl-moe-token`; diagnosis needs no GPU; reproduction needs 2 GPUs with at least 56 GiB free per card |
| **Evaluations** | issue |
| **Recorded runs** | 1 defective run, 150 steps; no control run bundled |

## Difficulty

`MegatronEngineWithLMHead` in `verl/workers/engine/megatron/transformer_impl.py` returns a two-item loss callback while the configuration enables per-token loss, CP2 and nonzero z-loss. MCore's final token normalization therefore receives no token count. Auxiliary gradients bypass manual policy-loss scaling, so changing the main loss scalar alone cannot fix the contract.

The framework, effective configuration, console log and scalar histories are retained. W&B has 148 records; console history reaches 150. At W&B steps 1 and 148, gradient norm is 223.222475206911 and 144.49191694520493, mean reward is 0.25 and 0, and mean response length is 233 and 1024. These observations do not isolate the defect's effect on training quality. The missing W&B tail is a disclosed logging gap.

## Reference solution

**issue**: identify the missing native `(loss sum, token count, metrics)` contract and explain how it skips final normalization of z-loss gradients. The common router prescaling backport and padding-mask compatibility fix are already present. The hidden reference contains the supplied repair patch; disabling z-loss is not a repair.

## Verification

An LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires at least 0.8. No solution tier is exposed: the provided material has no control curve or calibrated full-training threshold. No agent or judge evaluation has been run for this contribution.

The new image is **not yet built**. Static validation is recorded separately for review; image build, online asset preparation and GPU reproduction have not been verified here. Model weights, datasets, checkpoints and W&B binaries are omitted; prepare downloads pinned resources and checks all SHA-256 hashes and sizes. Framework and new glue use Apache-2.0, GSM8K uses MIT, and the model has its own license in `environment/model-LICENSE`.

The supplied directory contains no W&B binary or rollout sample dumps; scalar histories and stdout are retained without synthesizing missing artifacts. Console runtime addresses, ports and device labels are neutralized; scalar histories remain byte-identical.
