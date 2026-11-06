# Log-prob recompute labels miss the causal roll

Training Qwen2.5-1.5B-Instruct with GRPO on TACO EASY (8368 train / 184 test). `prepare_model_inputs` builds the log-prob recompute labels un-rolled instead of the causal `torch.roll(input_ids_rmpad, shifts=-1, dims=1)`, so every recomputed response log-prob is misaligned by one position — validation accuracy and reward collapse to zero while response length balloons.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Code RL |
| **Tags** | `off-by-one` `missing-causal-roll` `logprob-misalignment` `training-collapse` |
| **Framework** | verl (GRPO), FSDP actor + vLLM rollout, remove-padding path |
| **Run time** | ~6.3 hours |
| **Resources** | 8 GPU |

## Difficulty

The defect is in `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`: `input_ids_rmpad_rolled = input_ids_rmpad` instead of `torch.roll(input_ids_rmpad, shifts=-1, dims=1)`. The un-rolled labels are passed both to `logprobs_from_logits` — a plain log-softmax+gather with no internal shift, against standard `logits[i]`-predicts-`token[i+1]` — and to the fused kernel's `shift_labels`. This is the live FSDP/NO_PADDING path (`old_log_prob`, `ref_log_prob`, actor-update `log_prob`).

Nothing crashes: the run proceeds through all 400 steps. But the failure signature is loud once you know where to look: `rollout_corr/kl` holds ≈ 20–22 from step 1 (≈ 43 by step 399) where a healthy run sits near 0 — training and rollout assign massively different log-probabilities to the same tokens. Over the frozen run, `val-core/taco/acc/mean@1` falls 0.071 → 0.0 by step 150, `critic/rewards/mean` collapses to 0, and `response_length/mean` balloons from ~168 to ~2400 tokens. A 4-way check on Qwen2.5-1.5B confirmed that only `roll + no_padding_2_padding[-1]-slice` reproduces textbook log-probs (mean |Δ| = 0.0000); dropping the roll yields |Δ| ≈ 10 nats regardless of the padding slice.

Two nearby sites are traps, both restored to healthy upstream form: `verl/utils/torch_functional.py::log_probs_from_logits_response` is dead code in verl 0.9.0.dev (zero call sites), and `verl/workers/utils/padding.py::no_padding_2_padding`'s left-shift slice is only the *consumer* of the rolled array, not the defect — the real causal shift lives in the engine's `torch.roll`.

## Reference solution

- **issue**: a passing diagnosis returns an unhealthy verdict and names `prepare_model_inputs` and the dropped causal roll — the un-rolled labels fed to `logprobs_from_logits` and the fused `shift_labels`, misaligning every recomputed response log-prob by one position. Attributions that must fail: the `no_padding_2_padding` left-shift slice, the `taco.py` scorer, the parquet data or launch budget, and `log_probs_from_logits_response` (dead code on this engine).
- **solution**: restore `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`. Sufficient because the rest of the chain (`padding.py`'s slice, `torch_functional.py`) is already at healthy upstream form; a full 400-step run with the roll restored trains healthily (`rollout_corr/kl` ≈ 0, val peak 0.125).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent repairs the workspace and submits `report.md` without launching training; the harness runs the designated launch script for 200 steps (360-minute timeout) and scores the fresh logs on two components — a finite `val-core/taco/acc/mean@1` peak of at least **0.10** (weight 0.8) and mean `critic/rewards/mean` over training steps strictly above **0.10** (weight 0.2); pass requires both (`score == 1.0`). The final validation point and every training-reward step through the horizon must be present — partial or over-budget logs score 0. Calibration is on 2 runs per arm (8×H100, 400-step reference runs): repaired runs peak at 0.125 (step 150) / 0.114 (step 100) with reward mean ≈ 0.17–0.18 and `rollout_corr/kl` ≈ 0; the defective arm's best val is its step-0 baseline (0.071 frozen / 0.081 fresh) with 0.0 from step 200 and a 200-step mean reward of 0.082. The bars sit 0.019 above the defective best val and 0.014–0.025 below the repaired peak; margins only, no variance claim. The 200-step budget is the floor at which the reward criterion still separates the arms — the fresh defective run's mean reward stays above 0.10 until step ~165 — and repaired peak timing varies (step 100–250), so the budget is not cut below 200.
