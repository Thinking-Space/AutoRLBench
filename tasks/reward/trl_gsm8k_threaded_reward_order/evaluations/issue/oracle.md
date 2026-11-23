# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `workspace/trl/trainer/grpo_trainer.py::GRPOTrainer._call_reward_func_concurrently`
  - kind: `symbol`
  - observed: this method is a local addition to the vendored TRL 0.27.2
    tree. `_calculate_rewards` calls it for a synchronous reward function
    whenever `reward_num_workers > 1`, and `train_grpo.py` sets
    `reward_num_workers=32`. It cuts the batch into contiguous slices, one
    per completion at 32 completions per step, and submits one reward call
    per slice to a `ThreadPoolExecutor`. It then builds the result with
    `for future in as_completed(futures): rewards.extend(future.result())`.
    `as_completed` yields futures as they finish, not in submission order.
    Futures already done when it is called come out in arbitrary order, and
    the rest come out in the order their threads finish; each thread waits
    on a checker child process. So `rewards[i]` is often the score of some
    other completion of the step, from the same prompt group or another
    one. `_calculate_rewards` writes the list into `rewards_per_func` row by
    row, and everything downstream (group advantages, the loss, the logged
    per-completion rewards) treats row `i` as the reward of
    `completions[i]`.
  - expected: the slices' rewards concatenated in submission order, e.g.
    `for future in futures:` or `executor.map(...)`. Calling the reward
    function once on the whole batch (`reward_num_workers=1`) is also
    correct.

A passing diagnosis must name the ordering in the vendored TRL reward path
as the cause: `as_completed` returns the slices' results in completion
order, so the scores are attached to the wrong completions. It must also
connect this to GRPO. The batch still receives the right *multiset* of
scores, so the logged batch-mean reward and the user's checker look
correct. Per-group statistics (`reward_std`, `frac_reward_zero_std`) are
computed on misplaced scores, but nothing about their values gives that
away. GRPO's group-relative advantage then pushes up completions that were
not correct and pushes down completions that were. The policy gets little
or no signal toward correct answers or toward the required
`#### <number>` final line, and never learns them.

A diagnosis earns only partial root-cause credit, and cannot pass, if it
names only one of these:

- "the reward is wrong / noisy";
- "reward misalignment", without the ordering mechanism;
- the thread pool, `reward_num_workers` or the subprocess design, without
  the ordering bug.

Blaming the user's `correctness_reward` or `check_answer.py` for the
reordering is wrong. Those score each completion in order and correctly.

## Observable impact

- Primary source evaluation metric: `eval/gsm8k_accuracy`. It uses greedy
  decoding on the GSM8K test set (1,319 items), and an output cut off at
  the 1,024-token limit counts as wrong.
  - `workspace/results/eval_before.json`: 0.0167 at step 0.
  - `workspace/results/eval_after.json`: **0.0000** at step 200.
  - The share of outputs with a `#### <number>` final line falls from
    0.0235 to 0.0000.
  - A correctly wired run of this workspace reaches about 0.75.
- `workspace/results/metrics.jsonl` (200 steps, 4 prompts x 8 generations
  per step):
  - 25-step block means of the batch reward: 0.066, 0.025, 0.011, 0.014,
    0.009, 0.009, 0.003, 0.001.
  - `frac_reward_zero_std` rises from 0.61 to 0.99 and `grad_norm` falls
    from 0.0035 to about 0.
  - A run with the scores in order has a batch reward of 0.47 over steps
    1-25 and 0.76-0.85 in every later block.
- `workspace/results/rollouts.jsonl` logs every fourth completion (1,600
  rows) with its prompt text, the reward the trainer used and the
  advantage, but not the gold answer.
  - To re-check a row, join its prompt to the GSM8K training question to
    get the gold answer, then apply `common.extract`.
  - Re-checked, 33 rows are correct and 33 carry reward 1, but only one row
    is both: 32 correct rows carry reward 0 and 32 wrong rows carry
    reward 1.
  - 31 of the 33 reward-1 rows contain no `####` at all, and the checker
    can never score such a completion 1.
  - In the in-order control every logged reward matches the re-checked
    correctness.
- The process exits 0 (`train_grpo.py exit=0`) and nothing is logged as an
  error or warning about the reward.

## Trigger conditions

- `reward_num_workers > 1` with a reward function whose calls finish in an
  order that differs from submission order.
  - Here that is 32 single-completion slices, each blocking on a checker
    child process.
  - How many scores move depends on which thread finishes first, so the
    damage varies between runs with the same seed. In two other 12-core
    runs, one also stays at 0.000 test accuracy and one reaches 0.038.
  - It also depends on the host's CPU count. Buggy runs on 2 and 48 cores
    keep more scores in place and reach 0.56-0.58, against 0.72-0.75 for
    the in-order controls.
  - A diagnosis does not need to mention this, but it must not use it to
    argue that the code is correct.

## Symptoms, not root causes

- A batch reward that starts low and decays to about 0, and held-out
  accuracy going from 1.7% to 0%.
- `frac_reward_zero_std` near 1 and `grad_norm` near 0 at the end are
  consequences. Once the policy stops producing correct `####` answers,
  every group is all-zero and GRPO has nothing to learn from.
- `sampling/importance_sampling_ratio/mean` between 0.11 and 0.17 is TRL's
  default sequence-level vLLM importance-sampling correction with
  `top_p=0.95`. The in-order control shows values in the same range
  (0.14-0.39) and reaches 0.75 held-out accuracy, so it is not the cause.
- Rising completion length (about 227 to 300 tokens).

## Not the root cause

- **The user's reward function, `check_answer.py`, `common.extract` or the
  answer format.** `correctness_reward` returns one correct score per
  completion, in order. Re-checking any logged completion reproduces its
  true score; the scores are only attached to the wrong completions.
- **Learning rate (`1e-5`), `beta=0.0`, temperature 1.0, LoRA rank or the
  `dr_grpo` loss.** With the same values and in-order reward collection,
  the run learns the format and reaches 0.72-0.75 held-out accuracy.
- **"The task or format is too hard for the model" / "the reward is too
  sparse".** The base model already produces the `####` line on 2.3% of
  test items. With the rewards attached correctly, the batch reward is 0.47
  over steps 1-25 and above 0.76 afterwards.
- **vLLM importance-sampling settings, sampling `top_p`, the evaluation
  script or the 1,024-token limit.** These are the same in a healthy run of
  this workspace. Truncation is rare here: 0.5% of held-out outputs at
  step 200.
- **"Training did not run" or "the optimizer is broken".** Gradients are
  non-zero early on and the policy changes; it drifts away from the format
  instead of toward it.

## Evidence

A passing diagnosis should cite at least the following. Exact counts are
not required when the direction and rough size are right.

1. The `as_completed` concatenation in the vendored
   `_call_reward_func_concurrently` (reached through `reward_num_workers=32`)
   and why it reorders scores.
2. Evidence from `rollouts.jsonl` that correct completions were logged with
   reward 0 and/or incorrect ones with reward 1. Reward-1 completions
   without any `####` line count, and so does an equivalent re-check of the
   logged completions.
3. The held-out accuracy staying at about 0 (0.017 to 0.000), together with
   the batch reward decaying instead of rising.
