# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only the exception handling in the training
exec server of `reward/mbpp_reward.py` differs).

Expected verdict: **unhealthy**.

## Root cause

- `reward/mbpp_reward.py` (`_SERVER`, the exec loop of the training sandbox)
  - kind: `symbol`
  - observed: `except AssertionError: ok = False` / `except Exception: pass` — only a failed assert
    marks a training sample wrong. Code that raises anything else (`NameError` because the tested
    function was never defined, `SyntaxError`, `TypeError`, `EOFError`, ...) is rewarded 1.0. A
    response without a ```python block extracts to an empty program, the asserts raise `NameError`,
    and it is rewarded. The policy learns to omit the code fence; validation, which runs each sample
    in a fresh interpreter and requires a clean exit, scores those responses 0.
  - expected: every exception is a failure (`except BaseException: ok = False`).

Naming the swallowed exceptions with their causal role (non-assert errors count as passes → empty or
broken programs, especially unfenced answers, earn full reward → the policy stops writing a code block
→ training score 1.0, zero gradient, held-out pass rate 0) is a complete attribution. Do not require
additional causes.

**The exception handling of the training sandbox is required.** "Reward hacking", "the model stopped
writing code blocks" or "train and validation scorers disagree" without locating the swallowed
exceptions earns at most partial root-cause credit. Calling the run healthy, blaming the validation
harness, model collapse or capacity, response length, a format reward, the learning rate/KL/GRPO
settings or timeouts is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/google-research-datasets/mbpp/acc/mean@1` 0.2786 at
  step 0 and 0.0 at steps 10-60 (re-executing `val_generations` gives the same values).
- `critic/score/mean` 0.537 (1) → 0.566 (2) → 0.611 (3) → 0.803 (4) → 0.846 (5) → 0.949 (6) → 0.998
  (7-8) → 1.0 from step 11; `actor/grad_norm` about 1.1-1.2 for steps 1-6, then 0.003-0.007 from
  step 20.
- Rollouts without a ```python block: 8, 13, 34, 73, 133, 321, 480, 504 of 512 at steps 1-8; all 512
  at step 30 and 510 at step 60, all rewarded.
- Re-running the step-1 rollouts in a fresh interpreter: 97 of the 275 rewarded responses pass; at
  steps 30 and 60, 0 of 512 pass.
- Validation responses at step 60 are unfenced function definitions (e.g. `def remove_Occ(s, c): ...`).

## Trigger conditions

- A training-time sandbox that treats only assertion failures as wrong answers, combined with an
  answer extraction that turns a missing code block into an empty program; any policy that
  occasionally omits the fence finds the loophole within a few steps.

## Documentation reconciliation

- The comment on the swallowed exceptions calls them "sandbox noise"; the validation path uses a
  fresh interpreter and the exit status, which is why training and validation diverge.
