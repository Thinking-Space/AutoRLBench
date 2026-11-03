# Sealed environment oracle

## Verdict

**unhealthy.** The staged run reproduces the Agent-R1 ALFWorld multi-turn
tool-use GRPO recipe, but a tool-schema name mismatch has been injected into the
agent/environment tool interface.

## Root cause

`tool_schema_name_mismatch` — `tool_schema/alfworld_case/tools.py`. The tool
schema advertised to the policy (`tool_schemas()`) renames three executor tools
through `SCHEMA_NAMES = {name: name + "_object" for name in ("clean", "heat",
"cool")}`, so the model is told the tools are `clean_object`, `heat_object`,
`cool_object`. The execution registry `REGISTRY` and the dispatcher
`command_for` still only recognize `clean`, `heat`, `cool`. When the model calls
an advertised `*_object` name, `command_for` raises
`ToolCallError("unknown_tool")`, `episode.act` returns `"unknown_tool. The
environment state did not change."`, and `env.step` is never invoked for that
action. The failed call only consumes one of the 40 interaction turns and does
not reveal the correct name.

## Required remediation

Align the advertised tool names with the registered executor names: remove the
`_object` suffix so the schema advertises `clean` / `heat` / `cool` exactly as
`REGISTRY` and `command_for` expect. Do not rename the registry, and do not add
an alias layer that leaks the answer differently — the point is that the
advertised and dispatched names must be identical.

## Historical evidence

Evidence kind: `historical_run`. The frozen run is 63 steps with validation at
steps 0, 20, 40, 60, 63 (`test_freq=20`). Machine-readable copy:
`oracle/historical_evidence_manifest.json`. Agent-visible evidence lives under
`workspace/results/baseline63/` (`train.log`, `validation/*.jsonl`, three
`rollouts/*.jsonl` samples, and the offline W&B text logs).

Held-out unseen success (`val-aux/alfworld_valid_unseen/success/mean@1`):

| step | 0 | 20 | 40 | 60 | 63 |
| --- | --- | --- | --- | --- | --- |
| defective | 0.229 | 0.271 | 0.292 | 0.333 | 0.333 |
| control    | 0.458 | 0.646 | 0.708 | 0.771 | 0.792 |

The `pick_clean_then_place`, `pick_heat_then_place`, and `pick_cool_then_place`
families end at success rate 0 in the defective run.

## Why it is unhealthy

- `tool_schema/alfworld_case/tools.py`: the advertised schema names do not match
  the dispatched names for `clean` / `heat` / `cool`, so every episode requiring
  those tools is unsolvable.

The rest of the recipe is otherwise correct and must not be reported as the
cause:

- `tool_schema/alfworld_case/episode.py`: dispatches by registered name and
  returns `unknown_tool` for anything else — correct behavior given the names it
  is handed; the bug is the names, not the dispatcher.
- `tool_schema/alfworld_case/trajectory_flow.py` / `flow.py`: one training row
  per episode, assistant tokens masked 1 and tool/boundary tokens masked 0,
  append-only token histories, per-action token budget — all intact.
- `tool_schema/alfworld_case/train.sh`: standard GRPO (`adv_estimator=grpo`,
  `use_kl_loss=True kl_loss_coef=0.001`, `lr=1e-6`, FSDP2 fp32 main params,
  vLLM rollout, thinking disabled), 16 envs, 63 steps.

## Distractors that must not be reported as the defect

- The model **does** learn: overall unseen success rises 0.229 → 0.333.
  "Training did not improve" / "the policy failed to learn" is a wrong
  attribution.
- Not a JSON/tool-call parsing bug, not a reward change, not a system-prompt
  bug: the parameter schema, reward function, data, and the other tools
  (`look`, `inventory`, `go_to`, `open`, `close`, `examine`, `take`, `put`,
  `use`) are all intact.
- The step-0 capability gap (control already starts higher because clean/heat/cool
  work from the start) is expected, not a separate defect.

## Machine grading

Machine grading reads only the new experiment logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. The required held-out metric is
`val-aux/alfworld_valid_unseen/success/mean@1`, the unseen validation success
rate in `[0, 1]`. The instruction fixes `total_training_steps=63` and
`test_freq=20`, so the rerun has eval points 0, 20, 40, 60, 63.

`score_mode` is `best` (highest finite point over the run; a single eval over
48 unseen games is noisy, and the verl log interleaves per-progress `step:N`
lines so only the extracted value, not its step label, is reliable). That best
value must be at least `0.6` and at most `1.0`. Pass is score `1`; otherwise
`0`.

Both bounds are calibrated on the matched-schema control run — the same recipe
with the advertised names equal to the registered names, 63 steps, one run per
side:

- **Lower bound `0.6`.** The defective run's best is `0.333` and fails. An
  untrained matched-schema run sits at `0.458` at step 0 — also below `0.6`, so
  fixing the code without finishing the run does not pass. The control's best
  (`0.792` at step 63) clears the bound by `0.19` and crosses it by step 20
  (`0.646`).
- **Upper bound `1.0`.** Success is a rate, so `1.0` only rejects fabricated
  values; the control's best `0.792` is well under it.

`historical_final_baseline` in `grader_spec.json` is the *defective* run's
step-63 value (`0.333`): a repaired run must come in above it. Calibrated on one
run per side, so only margins are reported, not variance.
