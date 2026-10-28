# Merged LoRA never reaches the rollout engine

Training Qwen2.5-1.5B-Instruct with GRPO and a merged LoRA adapter on GSM8K. The rollout policy never receives the trained LoRA delta, so held-out accuracy stays at the base model's level while the actor keeps updating.

| | |
| --- | --- |
| **Author** | Jianing Yin |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `lora` `fsdp2` `weight-sync` |
| **Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Dataset** | openai/gsm8k main, 7473 train / 1319 test |
| **Framework** | verl @ `6f2aa24` with the pre-#7014 `transformer_impl.py`, GRPO, `run_grpo_lora.sh` |
| **Environment** | `verl` · 1 GPU |
| **Evaluations** | issue · solution |
| **Recorded runs** | 2 defective + 2 repaired runs, 60 steps each (seed 1 on 1 GPU, ~50 min; seed 2 on 2 GPUs, ~30 min) |

## Difficulty

`FSDPEngine.get_per_tensor_param` in `verl/workers/engine/fsdp/transformer_impl.py` handles `lora.merge=True` by calling `state_dict()` inside `merged_lora_context(...)`. Under FSDP2 that returns DTensors that alias the live parameters, but they are only gathered with `full_tensor()` by the generator the function returns, after the context has already copied the base weights back. vLLM therefore loads the base weights at every sync. This is the bug fixed upstream by verl PR #7014; the workspace is verl `6f2aa24` with that one file at its pre-fix revision `d45a68d5`.

Nothing fails. The actor trains (`actor/kl_loss` reaches 7.8, `actor/grad_norm` grows from 0.04 to 12.7), so it looks like an unstable run, and lowering the learning rate, raising the KL coefficient or adding importance-sampling correction are plausible but wrong. Held-out accuracy and training reward simply stay at the base model's level. The evidence that rules out "the optimizer diverged" is on the rollout side: on the same tokens, `rollout_corr/rollout_ppl` stays at the base value (1.50–1.59) for 60 steps while `rollout_corr/training_ppl` goes to infinity, so the policy that samples never changes.

| `val-core/openai/gsm8k/acc/mean@1` | step 0 | step 5 | step 30 | step 60 |
| --- | ---: | ---: | ---: | ---: |
| defective, seed 1 / 2 | 0.0857 / 0.0834 | 0.0781 / 0.0834 | 0.0819 / 0.0849 | 0.0842 / 0.0811 |
| repaired, seed 1 / 2 | 0.0811 / 0.0796 | 0.7286 / 0.7263 | 0.7703 / 0.7589 | 0.7771 / 0.7794 |

Two extra probe runs (`evaluations/issue/reference/weight-sync-probe.md`) record, for three LoRA target layers, what the actor should send (`W_base` and `dW = scaling * B @ A`) and what vLLM's `load_weights` actually receives, without changing either arm's code. The adapter's `dW` grows at every sync. Because the merge is done in bf16 and most per-element `dW` are below half a bf16 ulp, the comparison uses only the elements where `bf16(W_base + dW)` differs from `bf16(W_base)`, with exact bit equality and no tolerance. There the defective sync delivers `bf16(W_base)` at every sync (the whole tensor is bit-identical to the base weights), while the repaired sync delivers `bf16(W_base + dW)` on 98.98% of the elements. Of the remaining elements, 0.97% are exactly one bf16 ulp away and 0.05% two or more, consistent with the merge accumulating `B @ A * scaling` in a different order and precision than the fp32 reference that is rounded once.

## Reference solution

- **issue**: name the merged-LoRA weight sync in `get_per_tensor_param` (tensors materialized after `merged_lora_context` restores the base weights) as the root cause, and connect it to validation at the base level together with the growing rollout/actor divergence. Learning rate, KL coefficient, LoRA rank, bf16 merge precision, the GSM8K answer format, or "training did not improve" do not pass.
- **solution**: materialize every merged tensor while the context is still open, as the upstream patch does by yielding `full_tensor()` copies from inside `merged_lora_context`. Disabling `lora.merge`, LoRA or FSDP2 avoids the path instead of fixing it, and the rerun pins all three.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle and scores
  verdict, root cause and evidence. As implemented in the shared `judge.py`, a run passes only if
  the verdict score is 1.0 and the root-cause and evidence scores are both strictly greater than 0.8;
  the 0.2 / 0.6 / 0.2 weighted score is reported but does not decide the pass.
- **solution**: the agent submits `report.md` and does not train; the harness reruns the
  experiment per `run_spec.json` and grades only that run. `val-core/openai/gsm8k/acc/mean@1`
  at step 60 must be ≥ 0.4, and the verifier's own re-grade of the step-60 validation
  generations must agree with it within 0.002. Margins: 0.316 above the defective arm's
  step 60 (0.309 above its highest point, 0.091), 0.377 below the repaired arm's step 60,
  and 0.319 above the repaired arm's step 0. Calibrated on 2 runs per side.
