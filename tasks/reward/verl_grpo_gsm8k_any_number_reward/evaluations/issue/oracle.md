# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, config, data, seed and budget, only `rewards/gsm8k_reward.py` differs).

Expected verdict: **unhealthy**.

## Root cause

- `rewards/gsm8k_reward.py::compute_score`, loaded by `run_gsm8k_grpo.sh` through
  `reward.custom_reward_function.path/name` and used by verl for the training reward **and** for
  the logged validation metric `val-core/openai/gsm8k/acc/mean@1`
  - kind: `symbol`
  - observed: the function collects every number in the response (`_NUMBER.findall`, normalised for
    commas / `$` / trailing period) and returns 1.0 if the gold answer is in that set. It never
    looks for a final answer: a response that mentions the gold value as an intermediate quantity,
    a restated problem number or one of several candidates, and then ends with a different number
    or no answer at all, scores 1.0. The docstring explains why it was written (verl's strict
    `#### N` rule scored 0.0008 at step 0, "so the policy saw no signal"). Under GRPO the policy
    is pushed toward whatever raises P(gold number appears somewhere): more distinct numbers per
    response (9.4 → 14.3), longer step-by-step enumerations that recompute quantities and run into
    the 512-token limit (306 → 431 tokens mean, `response_length/clip_ratio` 0.10 → 0.43), and no
    commitment to a final line (`####` in 0.0% of rollouts from step 40). The logged score climbs
    from 0.54 to 0.81 and the logged validation "accuracy" from 0.594 to 0.684, while the policy's
    actual ability to answer falls.
  - expected: credit only a committed final answer in the format the prompt asks for, the last
    `#### <number>`, equal to the gold answer (the control's `rewards/gsm8k_reward.py`, equivalent to
    verl's built-in `openai/gsm8k` strict rule). Any reward that requires one final answer and
    compares only that answer is an acceptable remediation.

Naming this cause with the correct causal role (any-number match → credit without a committed
answer → GRPO optimises number coverage / hedging rather than correctness → inflated logged metrics,
no real learning) is a complete attribution. Do not require additional causes.

**The reward's leniency is required.** "The model never learned the `####` format" or "accuracy is
stuck" without connecting it to the reward that pays regardless of format is a symptom, worth at
most partial root-cause credit. Calling the run healthy because reward and validation accuracy rise
is wrong. Blaming the learning rate, KL coefficient, entropy, step budget, model size, the data,
or verl's reward-manager plumbing is wrong and gets no root-cause credit. Describing the rule as
"verl's flexible scoring" is also wrong: `flexible` takes the *last* number. This rule takes *any*
number.

## Observable impact

Every item in this section is derivable from the shipped workspace (`results/run100` plus
the verl fork). The counterfactual control-run figures are author-side reference and live
in Documentation reconciliation below.

- Logged held-out metric `val-core/openai/gsm8k/acc/mean@1` (defective reward): 0.594 (step 0) →
  0.617 (10) → 0.656 (30) → 0.679 (50) → 0.684 (100).
- The same validation dumps (`val_generations/{0,50,100}.jsonl`, 1,319 greedy responses) re-scored
  independently: strict (last `#### N` equals gold) defective 0.0015 → 0.000 from step 10 onward.
  **Format-free (last number in the response equals gold) defective 0.462 → 0.454 (10) → 0.429 (30)
  → 0.339 (50) → 0.244 (80) → 0.130 (100)**. Share of validation responses containing `####`:
  defective 0.022 → 0.004 (10) → 0.000 (40+). The defective policy's answers get worse
  under every reading except its own.
- Training rollouts (`rollouts/{1,50,100}.jsonl`, 1,024 per step): defective logged score 0.541 →
  0.655 (10) → 0.802 (30) → 0.734 (50) → 0.810 (100). Independently re-scored on the same
  rollouts: strict 0.013 → 0.000. Last-number 0.405 → 0.597 (30) → 0.437 (50) → 0.221 (100).
  **hedged credit (gold number present somewhere, last number wrong) 0.136 → 0.205 (30) →
  0.298 (50) → 0.589 (100)**. Distinct numbers per response 9.4 → 10.0 (30) → 14.3 (100). Mean
  response length grows: `response_length/mean` 306 → 431 tokens (equivalently 1,021 → 1,312
  characters — one item in either unit) and `response_length/clip_ratio` 0.096 → 0.435 (step 100).
- Step-100 defective responses are long numbered lists of "steps" that recompute the same
  quantities ("9. Total number of bolts ... 10. Total number of bolts ... 11. ...") or chain further
  arithmetic until the token limit, ending mid-computation with no committed final answer
  (43% end with an unfinished `\]` expression). Each additional number is another chance for the
  gold value to appear. A qualitative characterisation of these endings counts; the exact 43%
  is not required.
- Optimisation is healthy: `actor/grad_norm` 0.56 to 0.77, entropy 0.55 → 0.07, kl_loss
  0.0005 → 0.12, advantages bounded throughout. The defective run is not failing to
  optimise. It is optimising the wrong target.
- Timing: the defective run slows from 80 s/step to 100 s/step as responses lengthen.

## Trigger conditions

- A custom reward that tests containment of the gold answer anywhere in the response (or any
  extraction that does not isolate one final answer), used with a group-relative estimator. GSM8K's
  short integer answers and multi-step solutions make incidental matches common (14% of base-model
  responses already contain the gold number without ending on it).
- verl applies `reward.custom_reward_function` to validation as well, so the dashboard metric
  inherits the defect. Only an independent re-score of `val_generations/*.jsonl` reveals it.

## Documentation reconciliation

- Control reference (author-side; same code, config, data, seed and budget, only
  `rewards/gsm8k_reward.py` replaced by the strict last-`#### N` rule). Rebuttal context
  only — these figures are not shipped in the workspace and are not evidence items: the
  control logs validation 0.001 → 0.506 (10) → 0.563 (30) → 0.563 (50) → 0.557 (100),
  independent strict re-score 0.0008 → 0.557, format-free 0.462 → 0.557, `####` in
  0.029 → 0.975 (10) → 0.989 (100) of validation responses and 99.7% of rollouts; rollout
  score 0.019 → 0.479 (10) → 0.773 (100), last-number 0.396 → 0.773, hedged credit
  0.128 → 0.043, responses shorten to 240 tokens, clip_ratio 0.003, grad_norm 0.29 to 1.01.
- "The run is healthy, reward and validation accuracy go up" is wrong: both are the defective rule.
  Re-scored strictly the validation accuracy is 0.000. Format-free it falls from 0.462 to 0.130.
- "Sparse reward / the 0.5B model cannot learn the format" is wrong: the control, with the same
  model, data, prompt and budget, writes `#### N` in 97.5% of validation responses by step 10 and
  reaches 0.557 strict accuracy. The defective run stops writing `####` because nothing pays for it.
- "Learning rate / KL / entropy / step budget" are shared with the control and the defective run's
  gradients are healthy. No optimiser setting changes what the reward pays for.
- "This is just verl's `flexible` mode" is wrong (last number vs any number). The 14% hedged credit at
  step 0 and 59% at step 100 is exactly that difference.
- verl's reward manager, `custom_reward_function` loading, the GRPO estimator, uids and batch shapes
  are correct. Every logged score equals `compute_score` applied to the dumped response.
- `trainer.use_v1=False`, W&B offline mode, the numba/NumPy import warning and the critic-disabled
  warning are environment noise.
