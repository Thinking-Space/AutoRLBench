# Reference repair

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._compute_loss`: the DAPO denominator uses the token count of the entire generation batch without scaling it to the current gradient accumulation window. With steps_per_generation=4 and gradient_accumulation_steps=2, the loss and raw gradient for identical inputs are half the intended accumulation-window objective. During training multiply the denominator by current_gradient_accumulation_steps / steps_per_generation. Evaluation uses its unsplit batch denominator.

Apply `repair.patch` from the workspace root. The accompanying results are from one control run.

Both arms complete 24 optimizer steps. On nonzero captured objectives, the defective loss/reference ratio is approximately 0.5 and the control ratio is approximately 1.0. The independent objective comparison passes 12/24 optimizer steps in the defective run and 24/24 in the control. The held-out rewards are 0.1953125 and 0.20703125 respectively. This single paired run does not establish a general accuracy effect.

For every optimizer step, the verifier recomputes the DAPO objective from the recorded per-token probabilities, advantages and completion masks. Both microbatches must agree with the accumulation-window objective within relative tolerance 0.0005 and absolute tolerance 0.000002. All 24 optimizer steps must pass, with 48 records and at least one nonzero objective comparison.
