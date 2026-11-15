# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only the train-split filter in
`data_prep/gsm8k.py` differs).

Expected verdict: **unhealthy**.

## Root cause

- `data_prep/gsm8k.py::main` (train-split filter before `build_rows`)
  - kind: `symbol`
  - observed: `questions = [q for q in questions if len(q) <= MAX_QUESTION_CHARS]` filters only the
    question list; `answers` keeps all 7,473 entries, and `build_rows` pairs them with `zip`, which
    truncates silently. Every question after the first dropped one (raw row 175) is paired with a
    later problem's answer: 7,101 of 7,360 training rows (96.5%) have another problem's
    `reward_model.ground_truth` (and `extra_info.answer`).
  - expected: the same filter applied to both lists (or to the rows), so each question keeps its
    own answer.

Naming this one-sided filter with its causal role (positional pairing after the first dropped row →
training labels belong to other problems → correct responses are scored 0, training score near 0
→ no learning signal) is a complete attribution. Do not require additional causes.

**The filter on `questions` without `answers` is required.** "The training reward is near zero",
"the labels are wrong" or "the reward does not match the answers" without locating the one-sided
filter earns at most partial root-cause credit. Calling the data too hard, the reward function or
answer extraction broken, the format not learned, the learning rate/KL/GRPO settings wrong, or the
model too small is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` stays at 0.418-0.438 over
  steps 0-60; re-scored from `val_generations` (last number vs `gts`): 0.4488 (0), 0.4655 (10),
  0.4534 (20), 0.4579 (30), 0.4632 (40), 0.4519 (50), 0.4443 (60).
- `critic/score/mean` averages 0.017 (0.000-0.063) over 60 steps, far below the held-out accuracy of
  the same policy.
- In `rollouts/{1,30,60}.jsonl`, 677 of 1,536 responses answer their own question correctly (44%), but
  their `gts` belongs to another problem (e.g. Jessica/Rodney question, correct answer 15, `gts`
  3900, score 0).
- `data_prep.log` reports 7,360 training rows (the count is unaffected by the shift); rebuilding the
  parquet shows row 175 onward misaligned (row 175: sugar question, gold 2, `ground_truth` 58).
- Training still moves the policy on coincidental rewards: entropy 0.51 → 0.23, `actor/kl_loss` up to
  0.037, response length 303 → 358 → 318.

## Trigger conditions

- Parallel lists filtered independently and re-joined with `zip`, which never checks lengths.
- Any filter that drops at least one training row early; the test split is not filtered and stays
  correct, so validation alone does not reveal the defect.

## Documentation reconciliation

- The near-zero training score, the entropy drop and the flat held-out accuracy are symptoms of the
  same misaligned labels.
