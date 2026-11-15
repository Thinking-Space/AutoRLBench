# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only `reward/gsm8k_reward.py` differs: the
control calls verl's `flexible` GSM8K scorer directly, the defective workspace memoizes it).

Expected verdict: **unhealthy**.

## Root cause

- `reward/gsm8k_reward.py::compute_score` (`_SCORE_CACHE`)
  - kind: `symbol`
  - observed: scores are memoized under `key = (data_source, extra_info["split"], extra_info["index"])`.
    `index` is the prompt's row index, so the key is the same for all eight rollouts of a GRPO group
    and for every re-evaluation of a test prompt, and the reward worker (one, `reward.num_workers=1`)
    keeps the dict for the whole run. Every rollout therefore receives the score of the first scored
    rollout of its prompt.
  - expected: a score computed from each response (no memo, or a key that includes the response).

Naming this memo with its causal role (key without the response → identical scores within every
group → zero GRPO advantage → zero policy gradient, the policy never changes; the same memo replays
the step-0 validation scores) is a complete attribution. Do not require additional causes.

**The memo and its key are required.** "Advantages are zero", "the gradient vanishes" or "the run
does not learn" without tracing it to the reward memo earns at most partial root-cause credit.
Calling the run healthy because validation accuracy is stable, or blaming the learning rate, the KL
coefficient, the GRPO uid/grouping code, the advantage normalization, FSDP or numerical precision,
the GSM8K answer extraction, the data or the model's capacity, is wrong and gets no root-cause
credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`, which is exactly 0.4337 at
  steps 0, 10, 20, 30, 40, 50 and 60 — the step-0 scores replayed from the memo. Re-scoring the
  dumped generations (`val_generations/<step>.jsonl`, last number vs `gts`) gives 0.4655, 0.4632,
  0.4541, 0.4625, 0.4534, 0.4617, 0.4617: no improvement.
- The update is exactly zero for the whole run: `actor/pg_loss`, `actor/grad_norm` and
  `actor/kl_loss` are 0.0 at every step 1-60; entropy stays at 0.49-0.53 and response length at
  293-319 tokens.
- `critic/score/mean` moves between 0.27 and 0.41 without trend (the first rollout's correctness).
- Rollout dumps (`rollouts/1.jsonl`, `30.jsonl`, `60.jsonl`): all 192 groups have one score across
  their eight responses; 473 of 1,536 rollouts carry a score that contradicts their own final answer.

## Trigger conditions

- A reward-side memo whose key identifies the prompt but not the response, used with an algorithm
  that samples several responses per prompt (GRPO groups of 8) and a long-lived reward worker.
- verl does not check that rewards vary within a group; GRPO maps identical rewards to a zero
  advantage without any warning.

## Documentation reconciliation

- The frozen validation accuracy and the zero gradient are symptoms of the same memo, not separate
  defects. The training score's variation is resampling, not learning.
