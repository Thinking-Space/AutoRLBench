---
name: validate-task
description: >-
  Reviews an AutoRLBench task before submission. Checks the package against
  the harness schema, verifies that it contains a complete RL experiment and
  buggy-run evidence, and reports BLOCKER / MAJOR / MINOR findings with concrete
  adjustment suggestions. Use when asked to validate, review, audit, or check
  a task under tasks/ or a candidate task directory.
---

# validate-task

Validate one task directory at a time from the root of the AutoRLBench checkout.
Do not modify the task unless the user explicitly asks for fixes. The result is a
review report with a readiness verdict and actionable recommendations.

## Inputs

- Required: the task path, normally `tasks/<domain>/<task_id>/`.
- Optional: whether an expensive buggy-run reproduction is allowed. Do not run it
  by default; ask first because it may consume GPUs for hours.
- If the task path is omitted, ask the user for it. Do not guess among tasks.

## Workflow

1. **Run the packaged validator first.**

   ```bash
   python3 -m autorlbench validate <task-path>
   ```

   Treat every validator error as a `BLOCKER`, even if the task otherwise looks
   strong. Warnings require a task-specific judgement and become `MAJOR` when
   they weaken reproducibility or diagnosis.

2. **Inspect the package as a reviewer.** Read `task.toml`, `README.md`,
   `environment/prepare.sh`, every `evaluations/*/instruction.md`, every oracle,
   and the verifier implementation. Do not stop at the validator: it cannot tell
   whether the task is a meaningful, evidence-backed RL experiment.

3. **Inspect the experiment and frozen evidence.** Locate the declared training
   script and the workspace logs/W&B artifacts. Trace the launch script to the
   data, model, training loop, rollouts, reward, evaluation, checkpoint, and
   failure symptom. Check that the evidence can support the oracle without
   asking the reader to trust a claim.

4. **Optionally reproduce.** Only with user approval, run the submitter-side
   reproduction:

   ```bash
   python3 -m autorlbench rerun <task-path>
   ```

   A completed execution whose buggy outcome reappears is strong evidence. A
   non-reproduction is a `BLOCKER` unless the task README explains a verified
   external constraint; “it worked on my machine” is not sufficient.

5. **Write the report.** Use the format below, quote exact paths, and make every
   finding actionable. Prefer specific evidence over generic advice.

## Review Checklist

### 1. Package and metadata

- The directory is `tasks/<domain>/<task_id>/`; `domain` is one of
  `configuration`, `reward`, `data`, `environment`, `algorithm`, or
  `infrastructure`, and the task ID is stable, descriptive, and snake_case.
- Required files exist: `README.md`, `task.toml`, `workspace/`,
  `evaluations/issue/{instruction.md,oracle.md,verifier/test.sh}`, and
  `environment/prepare.sh` when remote assets are needed.
- `task.toml` and `README.md` agree on title, model, dataset, framework,
  algorithm, GPU count, run time, and failure description.
- `environment.image` is a digest-pinned image from `ENVIRONMENTS.md`, unless
  the task uses the documented exception and the README justifies it.
- `environment.gpus` is an integer count that the launch script actually needs.
  Reject GPU device IDs or unrelated host settings.
- No `AUTORLBENCH_*` variable is set by the task. Secrets, tokens, private
  credentials, and machine-specific absolute paths are absent.

### 2. Complete RL experiment

- The workspace contains a coherent training framework or a complete vendored
  framework snapshot, not a few isolated scripts assembled only to expose a bug.
- The declared `training.script` exists inside `workspace/`, is executable or
  invoked by an executable launcher, and is the single source of truth for the
  experiment.
- The launch path covers the complete experiment: environment and distributed
  initialization, input loading, model or policy construction, rollouts, reward
  computation, optimization, evaluation, and any checkpoint/restore behavior
  required to see the defect.
- Configuration and code are coupled. A script must load the staged model,
  datasets, hyperparameters, reward code, and data-processing code that the task
  claims to test; unrelated environment files are not evidence.
- Removing the defect must still leave a runnable baseline experiment. A task
  that only reproduces an exception in a synthetic helper is not an RL training
  task.
- The README states the intended training horizon and evaluation interval. The
  horizon must be long enough for the claimed behavioral failure to appear.

### 3. Buggy-run evidence

- `workspace/` contains the logs from the actual buggy run, in their natural
  output locations. At minimum, retain full or faithfully reconstructed
  stdout/stderr plus the decisive metric history.
- Keep offline W&B history when the source run used W&B, including enough files
  for the metric curve to be replayed; retain a couple of rollout samples when
  the defect concerns generation, tool use, reward computation, or data handling.
- Logs must be frozen evidence from the declared image, workspace state, seed,
  data, model, script, and GPU count. Logs imported from a materially different
  run are a `BLOCKER`.
- Logs contain enough identity information to connect them to the run:
  timestamps, command or framework version where available, evaluation step
  numbers, and the metric keys used by the README, oracle, and verifier.
- The failure symptom is visible in the logs and numerically supports the
  expected verdict. Do not rely only on a final assertion, screenshot, or prose
  summary.
- Avoid shipping checkpoints, model weights, and large raw datasets. If evidence
  was thinned or sanitized, record what was removed and why while preserving the
  values needed for diagnosis.
- A healthy-looking logged reward is not sufficient for task health. The evidence
  must include held-out or otherwise independent evaluation, plus rollout,
  optimization, truncation, or runtime signals when relevant.

### 4. Assets and reproducibility

- `prepare.sh` is the only networked stage. It fetches every required remote
  model, dataset, wheel, or source at a pinned revision with an allow list,
  hash, and byte size where applicable.
- Derived assets are rebuilt deterministically, or the non-determinism and
  fallback are documented and handled by the launch path.
- No model weights, checkpoints, tokenizers, or datasets are committed unless the
  packaging rules explicitly require shipping them.
- `prepare.sh` uses `set -euo pipefail` and portable paths, and does not mutate
  the read-only historical snapshot during evaluation.
- Environment variables and in-container paths match the launch script and task
  card exactly.

### 5. Root cause and issue tier

- The oracle gives one expected verdict and a concrete root cause, including all
  independent scored causes. If partial attribution earns partial credit, it says
  so explicitly.
- Root cause is tied to code, configuration, data processing, or reward logic in
  the defective workspace, with observed versus expected behavior. It does not
  merely restate the symptom.
- The failure has a plausible but non-trivial causal chain. The README identifies
  expected recipe settings and plausible distractors, and explains why they are
  not the root cause.
- Issue instruction, task card, and oracle agree on what may be inspected and
  what must be produced. The issue instruction forbids launching the full run,
  network access, and modifying the snapshot unless the task design explicitly
  says otherwise.
- The oracle and verifier do not leak the answer into `workspace/`, the issue
  instruction, or task metadata. Search the workspace for obvious answer-bearing
  patches, fixed files, and solution documents before allowing only verifier-side
  copies.

### 6. Verifier and scoring

- `evaluations/issue/verifier/test.sh` is safe to execute, exits with a meaningful
  status, and calls the documented judge or grading logic.
- A judge prompt requires the verdict, root cause, and evidence; weighted totals
  cannot let a wrong verdict pass.
- If a solution tier exists, its control comparison is clean: same data, seed,
  model, budget, environment, and everything except the claimed fix. Any other
  difference is a confounder and must be justified or removed.
- `run_spec.json`, `grader_spec.json`, the launch script, README, and oracle agree
  on script, arguments, env vars, final step, metric key, score mode, and pass
  threshold.
- The threshold separates both sides with margin and is calibrated against real
  evidence. The README states how many runs per side support it; one run per side
  is acceptable only when presented as such, not as a variance estimate.
- The grader validates the final step and rejects points outside the declared
  horizon; an earlier passing point must not satisfy the task.
- Verifier resources must be read-only or hashed. The verifier must not allow a
  submission to edit scoring code, datasets, or launch wiring and then pass.

### 7. Licensing, privacy, and hygiene

- Task and workspace licensing is stated and compatible with the upstream code.
- Datasets, prompts, rollouts, and logs contain no personal information or
  prohibited content unless properly licensed and documented.
- Generated artifacts, caches, virtual environments, `.git` directories, and OS
  metadata are absent from the packaged task.
- Large files are justified; check the repository diff and avoid accidental
  binary or cache additions.

## Finding Levels

- `BLOCKER`: the task cannot be accepted or reliably evaluated. Examples: schema
  errors, no complete experiment, missing or mismatched buggy logs, an answer
  leak, unverifiable root cause, non-reproducible assets, or a broken verifier.
- `MAJOR`: accepted only after revision. Examples: inconsistent metadata, weak
  evidence, an under-specified oracle, a solution threshold without enough
  calibration, or a confounded control.
- `MINOR`: clarity or hygiene issue that does not affect correctness.
- `QUESTION`: evidence is missing but might exist outside the package. Name the
  exact artifact to obtain and the test it must pass.

## Report Format

```markdown
# Task validation: <task-id>

## Verdict

**READY** | **NOT READY** | **READY WITH FOLLOW-UP**

## Summary

- Two or three sentences describing task quality and the decisive blocker, if any.

## Findings

### BLOCKER

- [path:line] Finding. Required adjustment.

### MAJOR

- [path:line] Finding. Recommended adjustment.

### MINOR

- [path:line] Finding. Recommended adjustment.

### QUESTIONS

- Question the contributor must answer, and why it matters.

## What to do next

1. Concrete change.
2. Validation to rerun.
3. Optional reproduction or review step.
```

Use `NOT READY` if any `BLOCKER` exists. Use `READY WITH FOLLOW-UP` only when no
`BLOCKER` exists and remaining `MAJOR` items are documented and approved by a
reviewer. Otherwise use `READY`.
