# Canonical solution oracle

## Root cause

- `run_gsm8k_grpo.sh`: `PPO_MINI_BATCH_SIZE=4` (`actor_rollout_ref.actor.ppo_mini_batch_size`) against
  `TRAIN_BATCH_SIZE=128`, `ROLLOUT_N=8`
  - kind: `config`
  - observed: 32 optimizer updates per rollout batch (verl multiplies the value by `rollout.n`), so
    most updates are off-policy: `ppo_kl` and `grad_norm` several times the control's, entropy rising
    to 0.73, truncation 0.37, training score 0.62 → 0.48, held-out accuracy 0.494 → 0.361 (control
    0.565).
  - expected: `PPO_MINI_BATCH_SIZE=32` (4 updates per rollout batch), the recipe's value.

## Required remediation

- Raise the mini-batch to the order of the rollout batch (`code_target/run_gsm8k_grpo.sh`, `fix.diff`:
  `PPO_MINI_BATCH_SIZE=32`). Any value that leaves a few updates per rollout batch without changing the
  rollout batch, `ppo_epochs` or the learning rate passes if the rerun clears the threshold. `8` (16
  updates, 0.541 in the recorded dose run) passes. `4` (0.361) does not. Lowering the learning rate to
  compensate for 32 updates, or shortening the run, is not the remediation and is unlikely to clear
  0.53. Memory settings (`ppo_micro_batch_size_per_gpu`, `use_dynamic_bsz`, `ppo_max_token_len_per_gpu`)
  are unaffected by the fix.

Machine grading reads only the new experiment logs. The grader re-scores the fresh run's
`experiment_logs/val_generations/100.jsonl` (1,319 greedy GSM8K test responses, at least 1,300 rows
required) with a fixed rule, the number inside the last `\boxed{}` numerically equal to `gts`, and
requires ≥ 0.53 (≤ 1.0). Pass is score 1, anything else 0. Logged metrics and symlinked dumps are
ignored. The logged `val-core/openai/gsm8k/acc/mean@1` is not used because verl 0.9.0's Math-Verify
scorer can silently under-report it by up to ~0.10 on a busy host (process-pool queue time counted
against the per-sample timeout). The `\boxed{}` rule agrees with the main-process Math-Verify re-score
within 0.03 at every recorded evaluation (it is stricter on truncated responses without a box).

## Recorded comparison (`recorded_comparison/`)

One run per arm, seed 42, same day, 1× RTX A6000 each (same node for all three). `control/run100/` is
the control (same workspace launched with the override `actor_rollout_ref.actor.ppo_mini_batch_size=32`,
equivalent to `fix.diff`). `dose_mb8/run100/` is a dose-response reference with `ppo_mini_batch_size=8`.
`RESULTS.md` is the side-by-side analysis (held-out per eval step under the logged, Math-Verify re-score
and last-number rules, training diagnostics for all three arms).

| held-out `\boxed{}` accuracy (grader rule) | step 0 | step 20 | step 50 | step 100 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| defective (mini-batch 4, 32 updates/step) | 0.482 | 0.494 | 0.456 | 0.361 | 0.494 (step 20) |
| dose (mini-batch 8, 16 updates/step) | 0.478 | 0.518 | 0.533 | 0.541 | 0.547 (step 70) |
| control (mini-batch 32, 4 updates/step) | 0.487 | 0.547 | 0.552 | 0.565 | 0.572 (step 70) |

Math-Verify re-score at step 100 for reference: 0.387 / 0.544 / 0.566.

Threshold 0.53: 0.036 above the defective run's best (0.494), 0.035 below the control at step 100
(0.565), above the control's step 0 (0.487), so applying the fix without rerunning fails. The dose arm
(0.541) passes by 0.011. Calibrated on one run per arm: these are margins, not a variance estimate.
Grader replay on the recorded dumps: defective exit 1 (0.361 < 0.53), control exit 0 (0.565).
