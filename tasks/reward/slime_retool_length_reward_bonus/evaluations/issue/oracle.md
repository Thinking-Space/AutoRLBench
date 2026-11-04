# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

`examples/retool/generate_with_retool.py::reward_func` — kind: `reward_hack`

After the math_dapo exact-match score and the bounded tool-call shaping term,
`reward_func` adds, for **every** sample, unconditionally:

```python
length_ratio = min(sample.response_length / args.rollout_max_response_len, 1.0)
result["score"] = result["score"] + 5 * length_ratio
```

with `rollout_max_response_len = 16384`. The bonus depends only on generation
length, not on correctness, so a wrong-but-long rollout can outscore a
correct-but-short one. Under GRPO the group-relative advantages then reward
longer generations.

- Observed: `result["score"] += 5 * min(sample.response_length /
  args.rollout_max_response_len, 1.0)` applied to all samples.
- Expected: reward depends only on answer correctness (math_dapo exact-match
  with `strict_box_verify=True`) plus the bounded tool-call shaping that nudges
  already-failing samples; generation length must not add reward.

## Observable impact

- Primary source evaluation metric: `eval/aime` (mean of `reward_func` scores
  on AIME 2024). Because the length term is inside `reward_func`, the logged
  `eval/aime` is length-polluted.
- Historical cutoff is train step 100. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 99.
- Logged `eval/aime` rises from about +0.82 at step 0 to +3.841 at step 99,
  whose late values exceed the healthy shaped-reward maximum `1` and are explained by
  `5 * length_ratio` (up to +5).
- Over the same window, eval response length grows from about 4.1k to 15.3k
  tokens and eval truncation rises from about 0.8% to 82.5%. Train response
  length at step 100 is about 15.2k with a 78.5% truncate ratio.
- Subtracting `5 * min(mean_effective_len / 16384, 1)` from logged `eval/aime`
  reconstructs a decontaminated score of about −0.417 at step 0 and −0.831 at step 99.
  Incorrect responses have score `min(-0.6, -1 + 0.05 * (tool_calls - 2))`,
  which ranges from -1.1 to -0.6 for nonnegative tool counts. Therefore
  `(score + 1) / 2` is neither exact accuracy nor a valid general upper bound.
  The conservative upper bound `(score + 1.1) / 2.1` is about 32.5% and 12.8%,
  respectively. No exact accuracy curve was recorded; a decline in these
  bounds alone does not prove an actual accuracy decline.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- The unconditional `5 * length_ratio` term in `reward_func` is present on
  every sample, including eval.
- GRPO then ranks longer completions higher, so length and truncation inflate
  through step 100.

## Documentation reconciliation

- The live defect coefficient is `5`, not `0.5`. A diagnosis that names the
  unconditional length bonus with the correct causal role is complete even if
  the numeric coefficient is approximate.
- The rest of the ReTool recipe (observation masking, token/log-prob
  alignment, context-budget clipping, GRPO hyperparameters, custom generate/rm
  hooks) is correct and must not be reported as the cause.
- `kl-loss-coef 0.00`, `--rotary-base 5000000`, `max_turns=16`, W&B debug
  logging, and `/root/…` asset paths are expected recipe properties, not
  defects.
- Rising logged `eval/aime` is a symptom of the polluted reward, not evidence
  of held-out health.
