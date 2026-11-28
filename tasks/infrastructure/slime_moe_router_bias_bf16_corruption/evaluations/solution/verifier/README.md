# Solution verifier

The verifier uses final-step mode at step 12. Passing requires `eval/correct_action_probability` in `[0.955, 1.0]`, `router/bias_dtype == torch.float32`, 8 unique router values, equal router/source hashes, and expert 7. The threshold is 0.93460 above the defective final value and 0.00393 below the control final value, and exceeds the control's step-0 value 0.95257. It was calibrated on one deterministic run per side, so no variance claim is made. `train.py` is hash-pinned and `report.md` is required.
