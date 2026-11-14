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
| **Run time** | ~3.5 hours |
| **Resources** | 8 GPU |

## Difficulty

The defect is in `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`: `input_ids_rmpad_rolled = input_ids_rmpad` instead of `torch.roll(input_ids_rmpad, shifts=-1, dims=1)`. The un-rolled labels are passed both to `logprobs_from_logits` — a plain log-softmax+gather with no internal shift, against standard `logits[i]`-predicts-`token[i+1]` — and to the fused kernel's `shift_labels`. This is the live FSDP/NO_PADDING path (`old_log_prob`, `ref_log_prob`, actor-update `log_prob`).

Nothing crashes: the run proceeds through all 200 steps. But the failure signature is loud once you know where to look: `rollout_corr/kl` holds ≈ 19–34 from step 1 (21.6 at step 1, 19.8 at step 200) where a healthy run sits near 0 — training and rollout assign massively different log-probabilities to the same tokens. Over the frozen run, `val-core/taco/acc/mean@1` peaks at 0.076 at step 50 and falls to 0.0 by step 150, `critic/rewards/mean` collapses from ~0.13 to 0, and `response_length/mean` balloons from ~168 to ~800 tokens. A 4-way check on Qwen2.5-1.5B confirmed that only `roll + no_padding_2_padding[-1]-slice` reproduces textbook log-probs (mean |Δ| = 0.0000); dropping the roll yields |Δ| ≈ 10 nats regardless of the padding slice.

Two nearby sites are traps, both restored to healthy upstream form: `verl/utils/torch_functional.py::log_probs_from_logits_response` is dead code in verl 0.9.0.dev (zero call sites), and `verl/workers/utils/padding.py::no_padding_2_padding`'s left-shift slice is only the *consumer* of the rolled array, not the defect — the real causal shift lives in the engine's `torch.roll`.

## Reference solution

- **issue**: a passing diagnosis returns an unhealthy verdict and names `prepare_model_inputs` and the dropped causal roll — the un-rolled labels fed to `logprobs_from_logits` and the fused `shift_labels`, misaligning every recomputed response log-prob by one position. Attributions that must fail: the `no_padding_2_padding` left-shift slice, the `taco.py` scorer, the parquet data or launch budget, and `log_probs_from_logits_response` (dead code on this engine).
- **solution**: restore `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`. Sufficient because the rest of the chain (`padding.py`'s slice, `torch_functional.py`) is already at healthy upstream form; a run with the roll restored trains healthily (`rollout_corr/kl` ≈ 0, val peak 0.114 at step 100 within the 200-step budget).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent repairs the workspace and submits `report.md` without launching training; the harness runs the designated launch script for 200 steps (360-minute timeout) and scores the fresh logs: the validation gate — a finite `val-core/taco/acc/mean@1` peak of at least **0.10** — decides the score whenever the fresh run logged any validation points; the mean `critic/rewards/mean` over training steps strictly above **0.10** applies only as a fallback for runs that produced no validation metric at all (`score == 1.0` on pass). Calibration is on one fresh run per arm at the 200-step budget (8-GPU): the repaired run peaks at 0.114 (step 100; 0.103 at step 200) with a steps-1–200 reward mean of 0.176 and `rollout_corr/kl` ≈ 0; the defective arm's best val is 0.076 (step 50) with 0.0 from step 150 on and a steps-1–200 reward mean of 0.065. The val bar sits 0.024 above the defective best val and 0.014 below the repaired peak, and the reward bar separates 0.065 from 0.176; margins only, no variance claim. The repaired run's peak timing varies (step 100 here; the same run left to continue peaks again at 0.125 at step 250), so the budget is not cut below 200.
