# Canonical solution oracle

The repair is the launch script's IcePop window override: `run.sh` appends
`actor.rejection_sampling.lower=0.67` / `actor.rejection_sampling.upper=1.5`,
narrowing the token-mask window that `examples/math/gsm8k_icepop.yaml` ships as
[0.5, 5]. The minimal fix removes the two override lines (restoring the shipped
window) or sets them back to 0.5 / 5 — see `fix.diff`. Any repair that returns
the behaviour-importance mask to a window at least as wide as the shipped one
while keeping the pinned files untouched is equivalent.

## Expected outcome on replay

With the window restored, the recorded control runs (same resolved
configuration otherwise, seeds 2 and 3) keep the masked token fraction at or
below 0.046 for all 87 steps, entropy below 0.23, and `eval-rollout/reward`
ends at 0.653 (seed 2) / 0.669 (seed 3), with 0.627/0.652 and 0.625/0.653 at
the first two epochs. The defective run ends at 0.5777 (seed 2; other narrow
seeds 0.380 and 0.507). `grader_spec.json` requires the final
(epoch-3) `eval-rollout/reward` ≥ **0.645**. The original gate was 0.62 (the
midpoint between the defective 0.5777 and the oracle 0.6687); the 2026-11-13
solution sweep recalibrated it after ten independent repairs separated into
two clusters: lr/clip mitigation-style repairs that leave the narrow window in
place landed at 0.620–0.632, while window restorations and oracle-equivalent
repairs landed at 0.658–0.685 (run-to-run σ ≈ 0.005–0.008). 0.645 is the
midpoint between the two clusters (≈0.01 error margin on either side, within
2σ of each cluster edge); 0.62 sits inside the mitigation cluster's range and
admitted finals as low as 0.620.

A replay that only shortens the run or freezes the policy cannot pass: the
grader requires all three epoch-end evaluations (min_eval_rows 3) and the
87-step completeness shape, and the data, model configuration, `areal/`
source, `examples/` entry points and lock files are sha256-pinned, so any
repair that games the metric by editing the framework, the reward logic, the
data or the model config fails stage integrity.
