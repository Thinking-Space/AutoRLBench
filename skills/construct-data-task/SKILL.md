---
name: construct-data-task
description: >-
  Constructs a data-domain AutoRLBench task from a detailed construction brief
  (a task 思路 / task_briefs document): purify a natively-defective pool from
  public sources with execution receipts, assemble a matched dirty/clean mix,
  probe the model's capability first, record buggy + control arms, and calibrate
  the grader. Use when the user asks to build or construct a task (or its data)
  from a brief, a construction idea, or a defect-family write-up — the packaging
  into tasks/<domain>/ is then finished with the package-task skill.
---

# construct-data-task

Turns one construction brief into one raw task under `raw_tasks/<task_id>/` with
recorded runs on both arms. The brief is the spec; the raw task is the
implementation plus its receipts. Packaging into `tasks/<domain>/` afterwards is
[package-task](../package-task/SKILL.md) — this skill ends where that one begins.

| Input | `raw_tasks/task_briefs/<family>.md` — anchors, mechanism, construction steps, judging evidence |
| --- | --- |
| Exemplar | `raw_tasks/verl_taco_wrong_expected_output/` (label-noise family; staging dir, moves to `tasks/` once packaged — for a finished data-domain task see `tasks/data/verl_taco_difficulty_mismatch`) |
| Output | `raw_tasks/<task_id>/` with `workspace/`, `environment/` (purification + assembly + manifests), `evaluations/`, and a recorded buggy run + control run used to calibrate the grader |

The core discipline: **the defect is native to the source data and is proven by
execution, never injected or hand-flipped.** Every construction claim that a row
is dirty must be backed by a receipt (which solution, which test, which
outputs) — reproducible from committed manifests plus pinned public sources.

## Workflow

### 0. Read the brief, extract the contract

Pull out and keep visible for the whole build:

- **Family and channel**: what exactly is wrong in the data, and whether it is
  false-negative (correct work punished), false-positive, or distribution
  mismatch. The channel determines what a "fingerprint" metric looks like later.
- **External anchors**: the papers / GitHub issues / audits quantifying the
  defect upstream (FNR percentages, known issue threads). These justify the
  family's existence in the README and set expected magnitudes.
- **Mechanism**: the causal story for how training on the defect shows up in
  curves (reward pinned, held-out regression, per-subset separation).
- **Falsifiable per-metric expectations** — write them down *now*, before any
  run, as TBD-marked claims. A calibration that contradicts them means the
  construction is wrong, not the expectation (see step 4's rejection rule).
- **The stack**: choose the training stack (framework, algorithm, reward
  function, launch script, hyperparameters) yourself to fit the brief and the
  defect's mechanism — there is no obligation to reuse an existing task's
  setup. Whatever you choose, keep it **identical between the buggy and control
  arms**: the injected data defect must be the only difference, which is what
  makes the diff clean and the control arm a valid baseline.

### 1. Purify the dirty pool (receipts or it didn't happen)

1. **Pool = set difference against an execution-verified reference** (e.g.
   problems whose bundled solutions pass their own tests vs the rest). Both
   sides must be public, pinned, and the difference reproducible; sanity-check
   the pool size against the brief's external anchor before going further.
2. **Execute bundled solutions in a disposable sandbox** to separate families
   inside the pool: `--network none` containers or rlimited subprocesses, loose
   (30–60 s) and strict (10 s) budgets, timeouts recorded per problem. Never
   execute untrusted dataset content any other way; never run interpreters from
   inside a download directory.
3. **Family criteria must be strict enough to be defensible.** State the
   load-bearing conditions explicitly (e.g. ≥2 textually distinct solutions
   agreeing on the same non-empty actual output on the same failing test, and
   no solution passing under the loose budget). Vacuous agreements (empty
   outputs, single-solution evidence) must be excluded *by rule*, and the rule
   written into the README — this is the difference between "wrong expected
   output" and an unrelated defect family.
4. **Emit a per-problem manifest**: problem id, failing test index, the
   dataset's expected output, the agreeing solutions' actual output, and any
   corrected reference the grader will later need. Commit the manifest (ids and
   keys); bulky receipts (full stdout dumps) stay outside the repo with a README
   pointer.
5. **Held-out test set**: verified-clean *by your own execution receipts*, drawn
   from the source's own test split, and disjoint from train. If the external
   verified reference covers only train, its cleanliness says nothing about
   your held-out split — verify it yourself.

### 2. Assemble the training data

- **Mix ratio from the brief's anchor** (e.g. 30–50% dirty), **matched on the
  confounder** (difficulty band) so the defect is the only difference between
  arms. Same seed for both arms; control arm = same assembler, clean manifest
  only, matched row count.
- **Deterministic assembly**: seeded sampling, stable shard order, sha256
  cross-checks on problem content (not just ids) so drifted upstream revisions
  fail loudly. Pin every source revision (40-char sha) in `prepare.sh` —
  dataset repos drift and "fetch main" is not reproducible.
- **Parquet byte hashes are platform-dependent** (pyarrow's snappy build differs
  per platform). Pin the linux x86_64 values — that is where the verifier runs;
  use a canonical content digest (rows → json → sha256) for cross-platform
  comparison during development.
- **Tokenizer-dependent metadata** (`prompt_tokens` columns) changes bytes when
  the model changes but not row membership; re-check that any token-length
  filter drops nothing when swapping models.
- Write an assembler `assemble_dataset.py` under `environment/` that takes
  manifests + mix spec + tokenizer and rebuilds the parquets; assert the
  expected hashes inside it.

### 3. Probe the model BEFORE recording anything

The channel a data defect exploits is only active if the model can actually
solve the problems — a model below the bar turns "wrong labels punish correct
solutions" into indistinguishable "data too hard". This is the single most
common way these tasks get built wrong and have to be redone.

1. Probe candidate models on the frozen held-out split **with the training
   validation's own sampling params** (same temperature / top_p / n) and the
   task's own reward function — not greedy, not a different judge.
2. **Capability bar** (from this build): sampled held-out accuracy ≥ ~30%.
   Below that, both arms move within noise and the defect is invisible.
3. **Truncation law**: measure P(correct | response truncated). If it is ~0,
   every truncated rollout is dead weight — sweep the response budget
   (4096 → 8192 → 16384) and set `MAX_RESPONSE_LENGTH` where truncation is
   negligible. Record the measurement in the README.
4. Model-form constraints: a reward that extracts the first fenced code block
   requires a non-thinking instruct model (thinking templates put scratchpad
   code in the block).
5. When the bar is not met by any available model, stop and report the probe
   table to the user instead of recording a doomed calibration.

### 4. Record both arms

- **Buggy arm**: the task-root `run_spec.json` (short: 50 steps, validation
  every 5, per-step rollout dumps). **Control arm**: same launch config, clean
  train parquet, matched size. The contrast between the two recorded runs is the
  diagnosis; one run alone proves nothing.
- **Pull protection**: incremental scp of every dump as it appears (dumps are
  hundreds of MB each; losing step 46 of 50 to a preemption hurts). Keep local
  copies even of partial runs.
- **Self-healing watchdog on preemptable infra**: poll for (ssh reachable) ×
  (run dir present) × (trainer alive) — ssh success alone is not liveness. On
  wipe: re-push workspace, re-assemble from persistent caches, re-verify hashes,
  relaunch. When the platform re-registers nodes, **probe each candidate node
  with a real connection** before pinning the ssh config — picking by list order
  wedges you onto the dead registration. Cap relaunches; log every cycle.
- **Rejection rule**: if the buggy arm's held-out curve *improves* within noise
  of the control, the calibration is invalid — do not tune thresholds around a
  channel that did not fire. Re-probe a stronger model and rebuild (this build:
  1.5B rejected at ~12% capability, rebuilt on a 4B instruct at 33.8%). Report
  the rejection honestly in the README; it is evidence about the defect family,
  not an embarrassment.
- Extract from each arm: the grader's fingerprint metric (per its definition),
  the training reward mean curve, and the held-out accuracy curve vs step-0.

### 5. Grader and calibration

- The solution grader needs an **external judge independent of the training
  reward** — the separation between the two scorers on the dirty subset (and
  only there) is the defect's fingerprint. For wrong-expected-output: re-execute
  each dumped score-0 response on the problem's recorded failing test and
  compare against the corrected output from the manifest.
- **Structural integrity checks before any metric**: byte-compare the launch
  script and reward function against pristine copies, sha-pin the test parquet,
  floor the train parquet row count, assert train/test disjointness, and take
  run-reality from the dump sequence itself (dumps 1..N present and no further).
- Metric = an aggregate over the **last K dumps** (e.g. max of the per-dump
  rate) so early-training noise cannot pass a late-stage criterion.
- Calibrate `historical_final_baseline` and the threshold from the two recorded
  arms: threshold strictly between the control's structural value and the
  defective arm's measured value, and above the control's step-0 value. State
  runs-per-side; with one run per side report margins only and say so.
- **`evaluations/solution/instruction.md` must match `run_spec.json` exactly**
  — command, env, steps, test frequency. The agent is graded on a run it never
  starts; any mismatch between what it is told and what runs is how a correct
  repair still scores 0.

### 6. Freeze evidence and hand off

- Slim dumps before freezing into `workspace/` (drop bulky fields like `gts`;
  keep three representative dumps plus all per-step metric logs), sanitize
  paths/labels, and record before/after shas in `environment/source_manifest.json`.
- Fill every TBD left in step 0 from the recorded runs; refresh the issue
  oracle's numbers from the buggy arm.
- Then run [package-task](../package-task/SKILL.md) for the packaging checklist
  and validation.

## Checklist

- [ ] Brief's external anchors verified against the purified pool's actual size
- [ ] Every dirty row backed by a committed manifest receipt; family criteria written down, vacuous agreements excluded by rule
- [ ] Held-out split verified clean by this task's own execution receipts, disjoint from train
- [ ] Both arms' parquets from one seeded assembler; control matched on confounder and size
- [ ] All source revisions pinned; assembly asserts linux x86_64 parquet hashes
- [ ] Model probe recorded (sampled accuracy at the training budget, truncation sweep) before any training run
- [ ] Buggy + control arms recorded; curves and fingerprint metric extracted; invalid calibrations rejected and documented, not tuned around
- [ ] Grader: external judge + structural integrity + last-K aggregation; thresholds calibrated with margins stated
- [ ] instruction.md ↔ run_spec.json byte-consistent on command/env/steps/frequency
- [ ] Evidence slimmed/sanitized with source_manifest receipts; all TBDs filled; package-task checklist queued
