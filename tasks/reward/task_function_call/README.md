# reward_design implementation diagnosis: task_function_call

## Case overview

- Evaluation key: `reward/task_function_call`
- Case family: `reward_design`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`
- Environment: `verl` · 8 GPUs

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

The staged `taco.py` ignores `fn_name` and forces every sample through the stdin path, so function-call problems always receive zero reward while stdin problems continue to score normally.

## Workspace and hidden evaluations


| Path                    | Visibility and purpose                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `workspace/`            | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/`    | Neutral diagnosis instruction, issue oracle, and task-local issue verifier.                |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier.           |


The agent cannot read oracle or verifier paths. `issue` grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

### Leaf-id sanitization

The original experiment name was `taco_qwen2_5_1.5b_function_call`, which
names the defect and so handed the agent the answer from the frozen evidence:
it appeared 30 times across the training log, the offline W&B
`config.yaml` / `output.log` / `debug.log` / `run-*.wandb`, and the hydra
configs under `outputs/`. Every occurrence inside `workspace/` is now
`__BENCH_TASK_ID__`, and the launch script reads
`EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`.
No harness code sets `BENCH_TASK_LEAF_ID`, so a fresh agent run emits exactly
the same string and the snapshot stays self-consistent. Do not restore the
original name. The binary `run-*.wandb` datastore was re-emitted record by
record (lengths, CRCs and 32KB block framing recomputed) rather than patched
in place; `oracle/task.yaml` still records the historical name because the
oracle tree is not agent-visible.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: reward_design
source_task: Benchmark_Evaluation/tasks/task_verl_coding/tasks_function_call
evidence_kind: injected_implementation_defect
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: true
target_files: verl/utils/reward_score/taco.py
validation_stack: TACO_EASY
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 8
  timeout_minutes: 660
  max_train_steps: 400
  eval_metric_key: val-core/taco/acc/mean@1
historical_baseline:
  actual_metric_key: val-core/taco/acc/mean@1
  eval_last: 0.05434782608695652
  eval_max: 0.08695652173913043
  n_eval_points: 9
  attempt: 1
```

## Original failure evidence

- dirty validation rises from 0.0217 to 0.0543 and peaks at 0.0870 at step 300
- official function solutions score 0 on both train and validation, while official stdin solutions still score 1
- the held-out set contains 41 call-based and 143 stdin problems, so every call-based row is capped at zero even for a correct function solution

A repaired run of this exact package was executed on reference devspace
`1236983` (`/user/user2/harbor-runs/task_function_call__stage__env`),
alongside a defective re-run on the same host, and the solution thresholds are
calibrated from those two curves — see `evaluations/solution/oracle/README.md`
for the table. The "healthy scorer reaches 0.139 at step 400" figure that
earlier revisions of this document used as the reference point belongs to
`tasks_coding_pass_01`, a different experiment whose stack included `pyext`;
this task's `prepare.sh` deliberately omits `pyext`, and the repaired run here
ends at 0.0978.

The preserved source metric is `val-core/taco/acc/mean@1`. The frozen workspace remains
the evidence package; solution grading requires a new 400-step training run and reads only
its experiment logs.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/utils/reward_score/taco.py::run_test`

Required fix targets:

- `verl/utils/reward_score/taco.py::run_test`: `restore`; target state/value: when `fn_name` is present, compile the candidate as call-based code, instantiate `Solution` when required, invoke `method(*inputs)`, and compare its return value; leave the stdin path unchanged.

Documentation reconciliation:

- The parquet data and prompts still identify call-based problems through `fn_name` and `Use Call-Based format`; the defect is confined to the scorer.
- The retained stdin path explains why aggregate validation remains nonzero despite all call-based rows failing.
- `prime_code` and agent-loop tool parsers are unused by the designated TACO launch and are not causal.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.
Ignoring `fn_name` and deleting the call-based execution path is the essential
root cause.

`solution` does not grade a structured diagnosis. The agent repairs the staged workspace and submits `report.md` without
launching training. The harness runs the designated launch script for 400 steps
and produces fresh experiment logs. Scoring is weighted:

- validation: the mean of `val-core/taco/acc/mean@1` over the pinned step set
  `200, 250, 300, 350, 400` must land in `[0.095, 0.35]` to score `1`,
  otherwise `0`. All five steps must be present, which is what the designated
  400-step run at `TEST_FREQ=50` produces.
- training reward: the mean `critic/rewards/mean` over training steps `1..400`
  must be strictly above `0.14` and at most `0.6` to score `1`, otherwise `0`.
  All 400 steps must be present.
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

Reference values from the calibration runs: repaired `0.1098` / `0.1838`,
defective `0.0783` / `0.1003`. Pinning the step set rather than taking a peak
matters because `TEST_FREQ` and `TOTAL_TRAINING_STEPS` are plain environment
overrides in the launch script, and because the repaired run's peak (`0.125`)
occurs exactly once. The upper bounds guard against a scorer rewritten to
return a constant reward, since both graded quantities come out of the file
the agent is asked to repair.

## External dataset asset

- Repository: `LIF1014/ptdbench-verl-coding-tasks-function-call-dataset`
- Publication status: `published`
- Pinned revision: `none — whole-repo fetch of the default branch`
- Dataset license: `not recorded in the task package`
- Provenance: Same processed TACO EASY snapshot as the other TACO verl tasks (8368 train / 184 test).


| Runtime path                            | Size              | SHA-256                                                            |
| --------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`      | 61,082,676 bytes  | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet`     | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |


`environment/prepare.sh` fetches the whole archive repo at the prepare stage
via the task-local `environment/fetch_hf.py` (cache-first under
`$AUTORLBENCH_MODELS`; no revision or checksum pins — the fetcher takes
whatever the repo's default branch serves), unpacks `workspace/taco_verl`
from `artifacts/workspace-assets.tar.gz`, moves it into the workspace, and
then verifies both parquet files against the digest and byte-count table
above. Nothing is baked into the task image.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Both the model and the dataset are fetched at the prepare stage by
  `environment/prepare.sh` (the only stage with network); they are not baked
  into the task image, and a backend without the prepare stage cannot hydrate
  them.
- The designated experiment uses 8 GPUs, a 400-step budget, and a large shared-memory allocation.
- Model weights and historical training checkpoints are not external dataset assets.
- `verify.timeout_minutes` is 660. The repaired reference run alone took 9h37m
  (the defective one 8h44m — restoring call-based execution makes the run
  slower), so the agent's diagnose-and-repair phase has under 1h20m of slack.

### Notes moved out of `environment/prepare.sh`

`prepare.sh` is mounted read-only into the agent container at
`/autorlbench/environment`, so it must not carry maintainer background that names
the defect, the sibling task, or the grading thresholds. These notes used to
live in its comments and belong here instead. Keep it that way.

**Migration.** `prepare.sh` replaced `environment/{Dockerfile,hydrate-assets.sh}`
on 2026-09-23. The legacy Dockerfile only installed a patched `pyext` and baked
`workspace/` into the image — it downloaded neither the model nor the dataset,
and neither is git-tracked under `workspace/`, so a container built from it
failed immediately on `MODEL_PATH` and `TRAIN_FILE`. The two assets were
reconstructed against the runtime paths, byte counts and SHA-256 values in the
"External dataset asset" table above.

**Sibling.** `reward_design/task_evaluator` is built from the same recipe — two
different injections into the same source file. Asset *contents* are
byte-identical, but each task publishes its own HF repo at its own revision, so
changes here usually need review there too. The two `prepare.sh` files are
deliberately **not** identical: only the sibling installs `pyext`.

**Why `pyext` is deliberately not installed.** The scorer tries
`from pyext import RuntimeModule` and falls back to a complete, self-contained
`_FallbackRuntimeModule` on `ImportError`; the fallback is reached on every
sample of the stdin path, so installing `pyext` or not is a choice that lands
on the reward, not a cosmetic one. The decision follows this task's own frozen
run manifest: `workspace/logs/wandb/run-20260821_153903-js8e27l6/files/requirements.txt`
is 282 lines with no `pyext`, while the sibling's frozen manifest is 283 lines
and records `pyext==0.7`. Sorted line-by-line, that one package is the only
difference. The shared image installs `pyext` for neither, which contradicts
the run recorded in the sibling, so the wandb manifests win, task by task. The two
implementations are not equivalent: `pyext`'s `from_string` execs into a
temporary globals dict and copies the new names into a fresh module, leaving a
function's `__globals__` and the module `__dict__` as two objects, whereas the
fallback execs straight into `module.__dict__`. For the single `getattr(mod,
'code')` use here the result is the same, but the semantics are not — which is
exactly why each task has to be aligned separately. To reinstate it, lift the
trailing section of the sibling's `prepare.sh` (installs into
`$AUTORLBENCH_PREPARED/pysite` and appends `PYTHONPATH` to `$AUTORLBENCH_PREPARED/env`).
Do **not** edit `docker/images/verl.Dockerfile`; that image is shared by every
verl task.

**Known residual leak.** The model now comes from the official
`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, which is
byte-identical on every allowed file to the former task snapshot
`LIF1014/ptdbench-verl-coding-tasks-function-call@8d94247`. The dataset repo is
still `LIF1014/ptdbench-verl-coding-tasks-function-call-dataset`. `prepare.sh`
must name it to fetch it, and the shared download cache (mounted read-only into
the agent stage) is keyed by repo name, so the string is visible to the agent
there too. Closing it means republishing the dataset under a neutral name and
updating `--repo` (and `--revision`, unless the repo is renamed in place).
