# Canonical issue oracle

Reconciled from the task's frozen evidence and the matched-schema control run.

Expected verdict: **unhealthy**.

## Root cause

- `tool_schema/alfworld_case/tools.py` (`SCHEMA_NAMES` / `tool_schemas`)
  - kind: `environment` (agent tool interface)
  - observed: the tool schema advertised to the policy renames three executor
    tools — `clean`, `heat`, `cool` — to `clean_object`, `heat_object`,
    `cool_object` (`SCHEMA_NAMES = {name: name + "_object" for name in
    ("clean", "heat", "cool")}`, applied in `tool_schemas()`). The execution
    registry `REGISTRY` and the dispatcher `command_for` still only recognize
    `clean` / `heat` / `cool`. The model is instructed to call the `_object`
    names, so `command_for` raises `ToolCallError("unknown_tool")`,
    `episode.act` returns the feedback `"unknown_tool. The environment state did
    not change."`, and `env.step` is never called for those actions.
  - expected: the advertised tool names match the registered executor names, so
    a `clean` / `heat` / `cool` call dispatches to the ALFWorld environment and
    advances it (the matched-schema control has no `_object` suffix).

## Observable impact

- Primary source evaluation metric: `val-aux/alfworld_valid_unseen/success/mean@1`
  (held-out unseen success; the sibling `valid_seen` behaves the same way).
- Held-out unseen success starts at 0.229 and plateaus around 0.271 → 0.292 →
  0.333 → 0.333 over steps 0/20/40/60/63, far below the matched-schema control
  (0.458 at step 0 rising to 0.792 at step 63).
- The three affected task families — `pick_clean_then_place`,
  `pick_heat_then_place`, `pick_cool_then_place` — end at success rate 0; the
  per-family breakdown in `results/baseline63/validation/*.jsonl` and
  `summary` evidence confirms it.
- Rollouts in `results/baseline63/rollouts/*.jsonl` show the assistant emitting
  `clean_object` / `heat_object` / `cool_object` tool calls answered by
  `unknown_tool. The environment state did not change.`, with those turns wasted
  until the 40-turn limit.

## Trigger conditions

- Any episode whose solution requires cleaning, heating, or cooling an object:
  the policy calls the advertised `*_object` name, `command_for` rejects it as
  `unknown_tool`, the environment does not step, and the failed call only
  consumes an interaction turn without revealing the correct name.
- Tasks that never need those three tools (plain placement, two-object
  placement, look-under-lamp) remain solvable, so the run still scores and still
  improves on the unaffected families.

## Documentation reconciliation

- The affected arm **does** learn: overall unseen success rises from 0.229 to
  0.333. "Training did not improve" or "the policy failed to learn" is a wrong
  attribution — the defect is the three mismatched tool names, and the residual
  gap versus the control is driven by the clean/heat/cool families being
  unreachable, not by a global training failure.
- This is not a JSON-parsing bug, not a reward change, and not a prompt bug: the
  system prompt, parameter schema, reward function, data, and the other tools
  (`look`, `inventory`, `go_to`, `open`, `close`, `examine`, `take`, `put`,
  `use`) are all intact and function normally.
- There is a pre-existing step-0 capability gap (the control already starts
  higher because clean/heat/cool work from the start), so the final gap cannot
  be attributed entirely to training dynamics.
