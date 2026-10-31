# Canonical solution oracle

## Root cause

- `rewards/gsm8k_reward.py::compute_score` (wired in with `reward.custom_reward_function`)
  - kind: `symbol`
  - observed: returns 1.0 whenever the normalised gold answer is among all numbers in the response.
    GRPO optimises number coverage and hedging instead of a correct final answer. The logged
    training score and validation metric (computed by the same function) rise while the strict and
    format-free accuracy of the responses fall.
  - expected: credit only a committed final answer, the last `#### <number>`, equal to the gold
    answer.

## Required remediation

- Replace the any-number match with a final-answer match (`code_target/rewards/gsm8k_reward.py`,
  `fix.diff`): extract the last `#### <number>`, normalise, compare to the gold answer. Deleting the
  custom reward (verl falls back to its built-in strict `openai/gsm8k` rule) is equivalent and also
  passes. A reward that keeps paying for the number anywhere, however it reports its own metric,
  does not pass, because the grader does not read the logged metric.

Machine grading reads only the new experiment logs. The grader re-scores the fresh run's
`experiment_logs/val_generations/100.jsonl` (1,319 greedy GSM8K test responses, at least 1,300 rows
required) with a fixed rule, last `#### <number>` numerically equal to the gold answer, and requires
≥ 0.28 (≤ 1.0). Pass is score 1, anything else 0. Logged metrics and symlinked dumps are ignored.

## Recorded comparison (`recorded_comparison/`)

One run per side, seed 42, 1× RTX A6000 each. `control/run100/` is the control run (same workspace
with `fix.diff` applied). `RESULTS.md` is the full side-by-side analysis (held-out per eval step
under the logged, strict, format-free and lenient rules, rollout hacking diagnostics, training
diagnostics).

| strict held-out accuracy (independent re-score) | step 0 | step 50 | step 100 | best |
| --- | ---: | ---: | ---: | ---: |
| defective | 0.0015 | 0.000 | 0.000 | 0.0015 (step 0) |
| control | 0.0008 | 0.563 | 0.557 | 0.569 (step 80) |

For reference, the defective run's own logged `val-core/openai/gsm8k/acc/mean@1` is 0.594 → 0.679 →
0.684 over the same steps, and its format-free accuracy falls 0.462 → 0.339 → 0.130.

Threshold 0.28: 0.279 above the defective run's best (0.0015), 0.277 below the control at step 100
(0.557), and above the control's step 0 (0.0008), so applying the fix without rerunning fails.
Calibrated on one run per side: these are margins, not a variance estimate. Grader replay on the
recorded dumps: defective exit 1 (0.000 < 0.28), control exit 0 (0.557).
