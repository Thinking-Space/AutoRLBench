# Repair notes (oracle)

## The defect

The staged experiment trains the actor with an FSDP2 learner forward while it
generates rollouts with a vLLM forward. With both sides at `bfloat16`
(`actor_rollout_ref.actor.dtype` and `actor_rollout_ref.rollout.dtype` in
`workspace/verl/verl/trainer/config/math_grpo.yaml`), the two forwards compute
materially different token probabilities for the same synchronized weights:
the frozen run's token-weighted mean absolute logprob gap is
0.0112136646 nat/token (1,026,663,841 response tokens), the max sampled-token
probability difference reaches 0.9999255938 (step 252), the max |Δlogprob| is
17.5484849 (step 326), the per-sequence ESS fraction collapses to 0.0989458,
and 5.64941% of sequences are mis-rejected (sequence log-ratio > log 3).
An FP16 arm of the same experiment measured 0.00135637 nat/token, ESS
0.9506284, mis-rejected 0.0000266335 — an 8.27x separation on the mean gap.

Root cause: BF16's 8-bit mantissa gives the two GPU computation paths
different rounding behavior; the learner forward and the generation forward do
not agree at BF16 effective precision despite identical nominal weights.

## The canonical repair

Both dtype keys flip `bfloat16` -> `float16` in
`workspace/verl/verl/trainer/config/math_grpo.yaml` (see `fix.patch`).
`float32` for both keys is an equally valid repair under the same reasoning.

The scoring replay measures the fix directly: 30 bounded training steps with
validation and checkpointing disabled, token-weighted mean of
`parity/logprob_abs_mean_nat` over those steps must be <= 0.005
(between the measured FP16 arm ~0.0014 and the BF16 arm ~0.0126 over the same
window; the frozen full-run BF16 value is 0.0112136646).

## What the grader accepts

- Any repair that makes the measured token-weighted mean fall under the bound
  with the pinned instrumentation pipeline in place -- the canonical dtype
  repair is the intended one; a code-level repair inside verl that genuinely
  makes the two forwards agree is equally valid.
- The effective configuration must match the task's contract for every key
  except the two dtype keys, which are free.
- Model and dataset files must match their content pins; the five pinned
  instrumentation files are restored before the run, so edits to them are
  neutralised (and a failed restore is caught by the same sha256 pins).

## Calibration arms (measured)

| arm | run | token-weighted mean | notes |
|---|---|---|---|
| BF16 (defect as staged) | frozen full run, 440 steps | 0.0112136646 | frozen evidence in workspace/logs/run_001 |
| BF16 (defect as staged) | bounded 30-step window of the frozen run | 0.0126181431 | first-30-steps projection of the scoring window |
| BF16 (defect as staged) | bounded 30-step replay (measured) | 0.0125929135 | beijing BATCH job 1539880, 2026-11-11; 0.2% off the projection |
| FP16 (canonical fix) | full run | 0.00135637 | README table |
| FP16 (canonical fix) | bounded 30-step replay (measured) | 0.0015140778 | beijing BATCH job 1539882, 2026-11-11 |

The 0.005 bound sits 3.3x above the measured FP16 replay arm and 3.4x below the
measured BF16 replay arm, so run-to-run noise cannot move either arm across it.

## Maintenance note

`verifier/resources/pristine/` holds byte-identical copies of the five pinned
workspace files. If any of those files ever changes in the task tree, re-copy
it into pristine/ and update the sha256 in
`verifier/resources/grader_spec.json` (`pinned_files`) in the same commit.
