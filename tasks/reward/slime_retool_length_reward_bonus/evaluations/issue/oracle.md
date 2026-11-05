# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

This run carries two independent defects; a complete attribution names both.

### 1. `examples/retool/generate_with_retool.py::reward_func` — kind: `reward_hack`

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

### 2. `examples/retool/generate_with_retool.py::postprocess_predictions` — kind: `tool_contract_mismatch`

The chat template's `TOOL_TEMPLATE` instructs the model to emit every tool
call as `{"name": <function-name>, "arguments": <args-json-object>}`, but
`postprocess_predictions` parses the matched tool-call JSON with
`arguments = tool_call_data.get("parameters", {})`. The `parameters` key
never occurs in the model's emitted call — it is the schema key of the tool
registry spec (`tool_sandbox.py::_register_default_tools`), which is what
makes the mismatch look superficially plausible. The parsed `arguments` is
therefore always the empty default.

- Observed: for `code_interpreter` calls, `code = arguments.get("code", "")`
  is always `""`, the `if code.strip():` guard fails, and no `("code", code)`
  is returned — `execute_predictions` then emits the invalid-action
  observation instead of any `<interpreter>` result. The `.get` default is
  silent, so the `except` arm never fires either.
- Expected: parse the arguments from the `"arguments"` key the template
  defines, so `code_interpreter` tool calls execute their code in the
  sandbox.

Naming both defects with their correct causal roles is a complete
attribution; naming only one of the two is partial. Do not require
additional causes beyond these two.

## Observable impact

- Primary source evaluation metric: `eval/aime` (mean of `reward_func` scores
  on AIME 2024). Because the length term is inside `reward_func`, the logged
  `eval/aime` is length-polluted.
- Logged `eval/aime` rises from about +0.356 at step 0 to +3.014 at step 69 —
  the frozen snapshot's last eval point (train step 70, eval interval 10) —
  which is outside the healthy math_dapo range `[-1, 1]` and is explained by
  `5 * length_ratio` (up to +5).
- Over the same window, eval response length grows from about 3.9k to 12.4k
  tokens and eval truncation rises from about 0.8% to 47.3%. Train rollout
  response length grows from about 2.8k at step 0 to about 9.2k at step 69,
  with truncation reaching 13.7%.
- Subtracting `5 * min(mean_effective_len / 16384, 1)` from logged `eval/aime`
  reconstructs a decontaminated score of about −0.839 at step 0 and −0.769 at
  step 69, bounding exact-match accuracy at no more than about 8.0% and 11.6%
  (`reward_func` floors an incorrect sample at `min(-0.6, …)` rather than
  `-1`, so these are ceilings, not point estimates): held-out accuracy stays
  near the floor while the polluted metric rises.
- The tool-contract defect is code-visible: model tool calls carry their
  payload under `arguments` (per the chat template) while the parser reads
  `parameters`, so template-conformant `code_interpreter` calls never reach
  `tool_registry.execute_tool` and the multi-turn code-execution loop cannot
  run; the `.get` default swallows the mismatch without raising.

## Trigger conditions

- The unconditional `5 * length_ratio` term in `reward_func` is present on
  every sample, including eval.
- GRPO then ranks longer completions higher, so length and truncation inflate
  through step 70.
- `postprocess_predictions` reads the tool-call argument dict from the
  `parameters` key while `TOOL_TEMPLATE` emits `arguments`, so
  template-conformant tool calls are rejected as invalid actions.

## Documentation reconciliation

- The live defect coefficient is `5`, not `0.5`. A diagnosis that names the
  unconditional length bonus with the correct causal role counts even if the
  numeric coefficient is approximate.
- The rest of the ReTool recipe (observation masking, token/log-prob
  alignment, context-budget clipping, GRPO hyperparameters, custom generate/rm
  hooks) is correct and must not be reported as a scored cause.
- `kl-loss-coef 0.00`, `--rotary-base 5000000`, `max_turns=16`, W&B debug
  logging, and `/root/…` asset paths are expected recipe properties, not
  defects.
- Rising logged `eval/aime` is a symptom of the polluted reward, not evidence
  of held-out health.
- A diagnosis that names both the unconditional length bonus and the
  tool-call key mismatch, each with its correct causal role, should receive
  full `root_cause_score`; naming only one of the two is partial.
