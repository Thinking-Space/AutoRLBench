# Canonical solution oracle

## Root cause

- `verl/experimental/agent_loop/agent_loop.py::AgentLoopWorker._compute_position_ids`
  - kind: `symbol`
  - observed: response position ids restart at 0 after the prompt window, so the actor and
    reference score rollouts under positions vLLM did not generate them with; held-out accuracy
    stays at 0.4511-0.4890 over two seeds
  - expected: one cumulative `compute_position_id_with_mask` over the concatenated attention mask

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`; `actor/kl_loss` 0.44 / 0.83
  at step 60 (seeds 42 / 43) vs 0.07 in the control.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `verl/experimental/agent_loop/agent_loop.py`: restore the single cumulative count (`fix.diff`).

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer, at
least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to cover steps 0-60, the
runner to match the verifier's copy, the regenerated training parquet to match the pinned raw GSM8K
train split, `data_prep/gsm8k.py` to be unmodified, and the held-out parquet to be the pinned raw
GSM8K test split verbatim.
