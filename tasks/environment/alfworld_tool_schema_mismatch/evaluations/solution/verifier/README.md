# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs
`val-aux/alfworld_valid_unseen/success/mean@1` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

## Which space the metric is read in

The held-out **unseen** validation success rate, in `[0, 1]` — the fraction of
the 48 `valid_unseen` ALFWorld games solved. It is logged once per eval in the
verl step-summary line (`val-aux/alfworld_valid_unseen/success/mean@1:<v>`).

This is the whole discriminator. The injected defect makes the three
clean/heat/cool task families unsolvable, so the unseen success rate cannot
climb past the ceiling set by the still-reachable families:

- defect present → unseen success plateaus near `0.333` (the historical run
  peaks at `0.333`).
- defect removed (advertised names aligned with the registry) → the three
  families become reachable again and unseen success climbs to `0.792` in the
  control.

So "did the agent align the tool names" is read directly off the metric.

## Bounds and how they are calibrated

The reference is the matched-schema control run — the same recipe with the
advertised tool names equal to the registered executor names, same image, same
assets, same 63-step budget, one run per side:

| step | 0 | 20 | 40 | 60 | 63 |
| --- | --- | --- | --- | --- | --- |
| control `valid_unseen/success` | 0.458 | 0.646 | 0.708 | 0.771 | **0.792** |
| defective `valid_unseen/success` | 0.229 | 0.271 | 0.292 | 0.333 | 0.333 |

`score_mode` is `best`: the verifier takes the highest finite value over the
run rather than the value at step 63. A single eval over 48 unseen games is
noisy, and the verl training log interleaves per-progress `step:N -` lines with
the eval-summary lines, so the extracted *value* is reliable but its *step
label* is not — `best` depends only on the value multiset and is robust to that.

- **Lower bound `0.6`.** The defective run's best is `0.333` and fails by
  `0.27`. An untrained matched-schema run sits at `0.458` at step 0, also below
  `0.6`, so fixing the code without finishing the run does not pass — the bound
  sits above the control's step-0 value as required. The control's best
  (`0.792`) clears it by `0.19` and the control crosses it by step 20
  (`0.646`).
- **Upper bound `1.0`.** Success is a rate; `1.0` only rejects fabricated
  values. The control's best `0.792` is well under it.

Verified against both real traces: the defective log scores `0`
(best `0.333` is below the `0.6` lower bound); the control log scores `1`
(best `0.792`).

## Calibration confidence

Calibrated on **one run per side** (the delivery completed a single seed). Only
margins are reported, not variance: best-to-lower-bound `+0.19` on the control,
`-0.27` on the defective run, and the control's step-0 value `0.458` is `0.14`
below the bound. Steps between the eval points (1–19, 21–39, …) and multi-seed
behavior were not verified.

## Accuracy space note

The metric is already a success *rate*, so no score→accuracy conversion is
involved. `valid_seen/success` moves the same way (0.5 → 0.854 control vs
0.229 → 0.292 defective) and could serve as a sibling check, but unseen is the
graded generalization metric.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
