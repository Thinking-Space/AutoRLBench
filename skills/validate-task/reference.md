# Task validation reference

This reference expands the review checklist with practical checks and evidence
expectations. It does not replace the packaged validator.

## Fast static commands

Run from the repository root and replace `<task-path>`:

```bash
# Schema, required paths, training script, image, and env types.
python3 -m autorlbench validate <task-path>

# Confirm the declared launch script and task card agree.
rg -n 'script|framework|algo|gpus|image|dataset|model' <task-path>/task.toml

# Find shipped logs and metric artifacts.
find <task-path>/workspace -type f \
  \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \) -print

# Look for accidental answer or solution material in the agent-visible tree.
find <task-path>/workspace -type f \
  \( -iname '*solution*' -o -iname '*oracle*' -o -iname '*fixed*' -o -iname '*answer*' \) -print

# Check large or generated files before submission.
find <task-path> -type f -size +10M -print

# Locate the newest submitter-side rerun output.
find outputs/rerun/<task_id> -mindepth 1 -maxdepth 1 -type d -print
```

Interpret these commands as starting points. A successful command is not a
successful validation; inspect the named files and their relationships.

## Rerun startup evidence

The required rerun output is `outputs/rerun/<task_id>/<timestamp>/`. Read
`command.sh`, `run.log`, `prepare.log`, and `rerun.log` directly. The check passes when
`prepare.sh` completes and `rerun.log` then shows at least two distinct non-initialization
training steps, with no image, dependency, model, or dataset error before them. An
intentional early stop is acceptable; a missing log, an initialization-only log, a
timeout, or a setup traceback is a `BLOCKER`.

Do not use this startup check to replace the frozen buggy-run evidence in `workspace/`.
It only proves that the shared image and `prepare.sh` can launch the experiment.

## Complete RL experiment

The central requirement is that the workspace be the defective side of a real
training experiment. A package fails this requirement when:

- It contains a small demo script plus files copied from a framework, but no
  launch path connects the model, data, training loop, and evaluation.
- The declared script invokes files or paths that are not in the workspace or
  rebuilt by `prepare.sh`.
- The bug is demonstrated only through unit-test snippets, synthetic tensors, or
  direct calls to a helper, rather than through training behavior.
- The training horizon is too short for the claimed failure mode.
- Evaluation is absent, cannot be found from the launch script, or does not
  measure the claimed symptom.
- The run depends on an undeclared GPU count, world size, distributed launcher,
  port, local model, or dataset.

Framework source may be vendored when the shared image does not contain it, but
the vendored tree should be the version used by the recorded run and should not
contain unrelated modified code unless documented.

## Evidence quality

Evidence is sufficient when an independent reviewer can:

1. Identify the run's framework, launch script, model, datasets, and GPU count.
2. Trace the metric key used by the task to code that computes it.
3. See multiple evaluation points covering the failure's development, not only
   a start and end claim.
4. Distinguish the root cause from a symptom and from expected recipe settings.
5. Repeat the grader's numeric extraction from the shipped or rerun log format.

For behavioral failures, include held-out performance plus at least one other
signal such as response length, truncation rate, tool-call success, loss, KL,
entropy, runtime, or rollout samples. For deterministic failures, preserve the
full traceback and enough preceding context to see the failing phase.

Do not use log prose alone as ground truth. If logs were reconstructed from
W&B, label the reconstructed file, retain the source format when practical, and
ensure the reconstruction is reproducible from the shipped W&B artifacts or
documented by the submitter.

## Review questions

- Does the README's failure claim appear in a log line with a step number?
- Does the declared final step match the log horizon and grader specification?
- Are metric names spelled identically in logs, README, oracle, and verifier?
- Can `prepare.sh` recreate the inputs without relying on contributor-local
  state?
- Would fixing only the oracle's root cause plausibly change the observed
  symptom?
- Does the task reward diagnosis or repair rather than brittle string matching?
- Does the solution grader accidentally accept a partial fix?
- Are all independent root causes required by the issue oracle?
- Is the failure significant enough that noise or a single seed cannot explain it?

## Common adjustments

| Finding | Adjustment |
| --- | --- |
| Only a few scripts reproduce a helper-level bug | Package the full framework and launch a complete training experiment that exposes the behavior. |
| No buggy-run logs | Re-run the experiment in the declared environment and package full stdout/stderr plus metric history. |
| Logs come from a different environment or workspace | Regenerate the evidence in the exact declared image and defective workspace. |
| No rerun output under `outputs/rerun/` | Run the submitter-side rerun until a few training steps appear, then revalidate. |
| `rerun.log` only shows startup banners | Fix the launch path or logging, rerun until real training steps appear, then revalidate. |
| Metadata and README disagree | Correct both and rerun the harness validator. |
| Root cause is only a symptom | Trace the symptom to code, config, data processing, or reward logic and rewrite the oracle. |
| Oracle has multiple causes without a partial-credit rule | State how many causes are required and how partial attribution is scored. |
| Solution threshold is tight | Add control runs, report the number of runs, and widen the acceptance band only with evidence. |
| Grader accepts early points | Require the declared final step and reject out-of-horizon evidence. |
| Verifier can be edited or bypassed | Move scoring assets out of the writable workspace or hash them in the grader. |
| Workspace leaks fixed code | Remove the fixed tree and preserve the control-side diff only in the solution oracle. |
