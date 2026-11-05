# Canonical solution oracle

## Root cause

The per-token callback in `verl/workers/engine/megatron/transformer_impl.py`
returns two values instead of the native loss sum / token count / metrics
contract. Under CP2 with nonzero MoE z-loss, final token normalization is
skipped for auxiliary gradients. Main policy-loss scaling cannot compensate
for auxiliary gradients that bypass the policy scalar.

## Required remediation

Restore the three-item callback with real routed-token counts, including
prompt tokens, and consistent global DP/CP policy scaling. The isolated image
already has the common router backport. `workspace-reference.patch` applies
to the delivered framework; `control/transformer_impl.py` is the exact
sanitized repaired source. Guards in the reference reject incompatible loss
aggregation and padding modes; the supplied token-mean/remove-padding run
does not enter those guards. Disabling z-loss or per-token mode is not a repair.

## Machine grading and calibration

Machine grading reads only new `experiment_logs` files and requires
`val-core/openai/gsm8k/acc/mean@1 >= 0.4` at step 150, with an accuracy cap
of 1.0. The 1.0 cap rejects invalid accuracy values. `score_mode=final`
rejects step-0-only and earlier-best logs.

The recorded run per arm plus a same-configuration rerun on the shared image
(2026-11-03) calibrate this draft threshold: defective validation is 0 at
steps 100 and 150 in both runs; repaired validation is 0.453125/0.546875
recorded and 0.515625/0.546875 rerun. Final margins are 0.4 above the
defective result and 0.146875 below the repaired result; the margin to the
lowest observed repaired point is 0.053125. The two runs share seed and
configuration but are not deterministic replays (step-1 grad_norm 223.2 vs
228.6), so the equal step-150 values are agreement between two executions,
not variance. These are observed margins, not variance bounds.
Neither run records step-0 validation, so the threshold's relation to the
initial checkpoint remains unverified. This contribution explicitly records
that calibration limitation and requires a final step-150 evaluation.

The source README discloses a W&B finish lifecycle difference at runtime;
the supplied sanitized workspaces differ in only the callback source file.
Logging completeness is not the scoring metric. Defective W&B ends at 148;
its actual console validation at 150 is retained separately. No missing
W&B points, rollout samples, or fresh training results are synthesized.
