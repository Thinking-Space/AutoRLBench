# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between
`3d274f6a7848a94d98d427ddda32aebfb9d5c19b` and
`9586c974bfbf8c9c6310c7615d320f42bfb59a82` (verl PR #5635) for
`verl/utils/reward_score/math_verify.py`; the rest of the workspace's verl
checkout is `6f2aa24abfc01cc72811b376aeb3936503885096`.

Expected verdict: **unhealthy**.

## Root cause

- `verl/utils/reward_score/math_verify.py::compute_score` (called by `reward/mixed_math_reward.py` for every MATH-style data source)
  - kind: `symbol`
  - observed: `compute_score` builds `math_verify.metric.math_metric(...)` and calls it
    inside `try: ... except Exception: pass`. math-verify's `parse()` wraps
    extraction in its signal-based timeout, which installs a `SIGALRM` handler
    with `signal.signal()` (`math_verify/utils.py`). The reward loop runs the
    custom reward synchronously through `loop.run_in_executor`
    (`verl/experimental/reward_loop/reward_manager/naive.py::NaiveRewardManager.run_single`),
    i.e. in a worker thread, where `signal.signal()` raises
    `ValueError: signal only works in main thread of the main interpreter`;
    math-verify re-raises it as `ValueError: Math-Verify 'parse' function doesn't
    support threaded environment ...`, and the bare `except Exception: pass`
    turns it into a score of 0.0 for every MATH-500 and MATH-lighteval sample.
  - expected: scoring that does not depend on the signal-based timeout when it
    runs in a worker thread (e.g. `parse(..., parsing_timeout=None)` and
    `verify(..., timeout_seconds=None)` as in the upstream fix, or a subprocess),
    and no silent conversion of a grader failure into a zero reward.

## Observable impact

- Primary source evaluation metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
- It is exactly 0.0 at all 21 validation points (steps 0 to 100), including
  step 0, before any update. Re-grading the same validation generations offline
  in the main thread with math-verify gives 0.5634 at step 0, 0.5433 at step 50
  and 0.5272 at step 100.
- The run is otherwise healthy-looking: `val-core/openai/gsm8k/acc/mean@1`
  rises from 0.0811 (step 0) to 0.7415 (step 50) and 0.7657 (step 100), and
  training reward `critic/score/mean` rises from 0.068 (step 1) to 0.426
  (step 50) and 0.424 (step 100).
- In the training rollouts the GSM8K half carries all of the reward (strict
  score 0.141 at step 1, 0.852 at step 50) while every MATH sample logs 0.0,
  although 39.4% / 53.5% / 46.5% of them are correct at steps 1 / 50 / 100 by an
  offline main-thread re-grade. MATH groups therefore contribute zero GRPO
  advantage.
- Frozen evidence is in `outputs/train.log`, `outputs/console-history.jsonl`,
  `outputs/wandb-history.jsonl`, `outputs/rollouts/` and
  `outputs/val_generations/`.

## Trigger conditions

- A custom reward that calls `verl.utils.reward_score.math_verify.compute_score`
  (here for every data source other than `openai/gsm8k`).
- verl's reward loop with the default `naive` reward manager, which runs a
  synchronous `compute_score` through `loop.run_in_executor`, so every call is
  off the main thread.
- math-verify 0.9.0, whose `parse()` timeout uses `signal.signal(SIGALRM)`.

## Documentation reconciliation

- "Training did not improve" is the wrong attribution: GSM8K accuracy and
  training reward rise normally; only the MATH reward is missing.
- The model can solve MATH (about 55% of MATH-500 at step 0 by an offline
  re-grade), the answers use `\boxed{}` as instructed, and responses are
  almost never truncated (`response_length/clip_ratio` at most 0.0059 at any
  step), so model capability, answer format and `max_response_length` are not
  the cause.
- math-verify is installed and works: calling the same `compute_score` from the
  main thread on the same answers scores them correctly. The failure exists
  only in the reward loop's worker threads.
- The reward router `reward/mixed_math_reward.py` routes correctly; the zero
  comes from the scorer it calls.
- Lower MATH accuracy late in the run is not established as a training effect:
  across two seeds the repaired recipe's step-100 MATH-500 gap was +5.0 points
  (p=0.007) and +0.8 points (p=0.74). A stable acceptance threshold based on
  MATH learning gain has not been established; the defect is in the grading path.
