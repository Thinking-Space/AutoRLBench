# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only the type of `reward_model.ground_truth` in
`data_prep/gsm8k.py` differs).

Expected verdict: **unhealthy**.

## Root cause

- `data_prep/gsm8k.py::build_rows` (`"ground_truth": int(float(extract_solution(answer)))`)
  - kind: `symbol`
  - observed: the label is stored as an integer. `reward/gsm8k_reward.py` calls
    `verl/utils/reward_score/gsm8k.py::compute_score`, which extracts the response's answer as a
    string and tests `answer == ground_truth`; a string never equals an int, so every reward — in
    training and in validation — is 0.0.
  - expected: the string label (`extract_solution(answer)`), as the scorer expects (or a numeric
    comparison in the reward).

Naming the integer cast with its causal role against the scorer's string equality (every reward 0 →
all-zero groups → zero advantage and gradient → the policy never changes; logged accuracy 0.0 is the
same mismatch) is a complete attribution. Do not require additional causes.

**The int label vs string comparison is required.** "The reward is always zero" or "the model gets
nothing right" without locating the type mismatch earns at most partial root-cause credit. Blaming
the model's capacity, the `####` format, the extraction regex, the learning rate/KL/GRPO settings or
vLLM is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` is 0.0 at steps 0-60, while
  re-scoring `val_generations` (last number vs `gts`) gives 0.4602 (0), 0.4632 (10), 0.4549 (20),
  0.4632 (30), 0.4564 (40), 0.4701 (50), 0.4602 (60).
- `critic/score/mean`, `actor/pg_loss`, `actor/grad_norm` and `actor/kl_loss` are exactly 0.0 at every
  step 1-60; entropy 0.46-0.56 and response length 290-326 tokens without trend.
- In `rollouts/{1,30,60}.jsonl`, 648 of 1,536 responses end with the correct number and are scored
  0.0; `gts` is a JSON integer in every dump (e.g. `"gts": 21`).

## Trigger conditions

- Any label whose Python type differs from what the rule-based scorer compares against; verl's
  GSM8K scorer uses `==` on strings and returns the format score (0.0) on a mismatch, without any
  warning.

## Documentation reconciliation

- The zero training score, zero gradients and zero logged validation accuracy are one symptom of the
  type mismatch; the model's real accuracy is about 0.46 throughout.
