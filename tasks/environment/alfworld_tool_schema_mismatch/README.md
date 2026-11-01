# ALFWorld tool-schema name mismatch

Training Qwen3-4B with GRPO on ALFWorld tool-use tasks. The tool schema
advertised to the policy renames the clean/heat/cool tools to
clean_object/heat_object/cool_object, but the environment still only dispatches
clean/heat/cool, so those calls return `unknown_tool` and never advance the
environment.

| | |
| --- | --- |
| **Author** | cyh2004 |
| **Domain** | environment |
| **Model** | Qwen3-4B |
| **Dataset** | ALFWorld, difficulty-v1-s42 train (636 unique tasks, 1008 stratified draws) / balanced1008-s42 validation (96 games: 48 seen + 48 unseen, 16 per family) |
| **Framework** | agent-r1 (verl 0.7.0 GRPO), `tool_schema/alfworld_case/train.sh` |
| **Environment** | `verl` · 4 GPUs (the intended `agent-r1-alfworld` image is not yet published; solution-level retraining on the `verl` image is **unverified** — issue level is read-only diagnosis and works on any image) |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control, 63 steps each (eval at 0/20/40/60/63), single seed |

The datasets are existing runtime assets, not absent task inputs. The original
AutoRLBench metadata declares `difficulty-v1-s42` and `balanced1008-s42`, and the
frozen `workspace/results/baseline63/train.log` records loading 1008 training
rows and 96 validation rows. The parquet files are not embedded in this local
workspace; their absence here does not justify removing the solution tier.
The runtime must expose the existing data at
`/workspace/tool_schema/datasets/alfworld/difficulty-v1-s42/train.parquet` and
`/workspace/tool_schema/datasets/alfworld/balanced1008-s42/validation.parquet`,
together with the games referenced by those rows. The local prepare script
only fetches the model; the deployment's data staging/mount connection needs
to be verified separately. No fresh end-to-end dataset mount was tested here.

## Difficulty

The defect is in `tool_schema/alfworld_case/tools.py`. `tool_schemas()` builds
the schema advertised to the model, but applies
`SCHEMA_NAMES = {name: name + "_object" for name in ("clean", "heat", "cool")}`,
so the model is told the tools are `clean_object` / `heat_object` /
`cool_object`. The execution registry `REGISTRY` and the dispatcher
`command_for` still recognize only `clean` / `heat` / `cool`; `episode.py` calls
`command_for(name, …)` by the advertised name, which raises
`ToolCallError("unknown_tool")` and returns `"unknown_tool. The environment
state did not change."` without stepping the environment.

Nothing fails loudly: the JSON tool-call shapes are valid, the system prompt,
reward function, data, and the other nine tools are intact, and the run still
learns on the unaffected task families — overall held-out unseen success rises
from 0.229 to 0.333. The single strongest piece of evidence that rules out "the
policy failed to learn" is the per-family breakdown: `pick_clean_then_place`,
`pick_heat_then_place`, and `pick_cool_then_place` end at exactly 0, while the
other families improve, and the rollouts show `*_object` calls answered by
`unknown_tool`.

| held-out unseen success | step 0 | 20 | 40 | 60 | 63 |
| --- | --- | --- | --- | --- | --- |
| defective | 0.229 | 0.271 | 0.292 | 0.333 | 0.333 |
| control | 0.458 | 0.646 | 0.708 | 0.771 | 0.792 |

## Reference solution

- **issue**: a passing diagnosis must call the run unhealthy and name the
  tool-schema name mismatch — the advertised `clean_object`/`heat_object`/
  `cool_object` names versus the registered `clean`/`heat`/`cool`, which makes
  `command_for` return `unknown_tool` so the environment does not advance.
  Attributing it to "training did not improve", a JSON/parsing bug, the reward,
  the prompt, or any benign setting must fail.
- **solution**: remove the `_object` suffix so the advertised names equal the
  registered executor names (`SCHEMA_NAMES = {}`). See
  `evaluations/solution/oracle/reference_fix.patch`. That is sufficient because
  the registry and dispatcher are already correct; only the advertised names are
  wrong.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict_score 1.0 with root_cause_score and evidence_score each above 0.8.
- **solution**: metric-only on `val-aux/alfworld_valid_unseen/success/mean@1`
  (held-out unseen success rate), `score_mode: best` over the fixed 63-step
  rerun. Pass is a best value in `[0.6, 1.0]`. The defective run's best is 0.333
  (fails by 0.27); the control's best is 0.792 (clears by 0.19); the control's
  step-0 value 0.458 is 0.14 below the bound, so a step-0-only submission does not pass. Calibrated on one run per side, so only margins are
  reported, not variance.

The grader requires validation at the declared final step and rejects evaluation
points outside the declared horizon before applying the existing best-score gate.
A passing earlier point alone does not establish completion.
