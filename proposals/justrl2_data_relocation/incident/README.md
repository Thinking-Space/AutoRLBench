# Checkpoint incident continuation

This revision turns the diagnosis into an already-materialized incident. The
submitted surface is `loader.py`; the bench retains the sampling plan, data,
model, generation/reward/training pipeline, and outcome instrumentation.

The authoring stage runs original steps 0–3 with the eight-step schedule, preserves
the complete actor and critic state, and copies it out of the rolling checkpoint
directory. Scene assembly physically migrates difficult payloads but leaves their
manifest paths stale. A current catalog makes the new locations discoverable.
Neither assembling a scene nor grading a repair trains an LLM agent.

The benchmark starts the solver with `incident/dataset` already in that state,
`incident/checkpoints` at step 3, the original logs, and the defective resolver.
The harness then resumes steps 4–7 from the saved checkpoint and consumes only
the submitted resolver. Missing-data retries are allowed; replacing a requested
logical sample with a different one fails the exact ID/content check.

Acceptance requires actual model/Adam/scheduler/RNG restore, counters 4→8, four
real updates per role, exact scheduled logical sample groups, final checkpoint
readback, fixed evaluation completion and unchanged sealed input. Reward and
hard-sample proportions are observations, not acceptance thresholds.

## Authoring and use

Inside the pinned public runtime:

```bash
python3 incident/build.py --prepared /work/prepared --output /work/authoring
python3 incident/scene.py verify /work/authoring/scene
```

`build.py` is a one-time authoring operation; evaluation must reuse the artifact.
The archived scene inventory and snapshot pin bind the exact checkpoint. The
task's setup materializes it before agent entry. `run.py` is the trusted
continuation driver; `verify_resume.py` independently audits resulting files.

`loader.py` returns the original manifest path and reproduces the failure.
`reference_loader.py` is maintainer-only: it queries the current catalog by
logical ID and content digest. It is not part of the solver workspace.

The checkpoint is a generated training artifact distributed separately from Git.
The supported delivery mode is a pinned local scene directory, supplied either
through `RELOCATION_SCENE_SOURCE` during preparation or through the standard
AutoRLBench dataset cache at `incidents/<scene_sha256>`. The task README gives
the `--datasets-dir` installation route. No public checkpoint URL is claimed.

## Measured evidence

Native reference run 1430316 restored iteration 3 and completed updates
4–7. Its fresh acceptance receipt, complete state observations, and curves are
under [evidence/](evidence/README.md). Actor and critic both restore Adam counter 4,
advance through counters 5–8, and pass final checkpoint 7 readback. The fixed final
evaluation contains 16 examples; its observed mean reward is 0.375, which is not
a pass threshold.

The actual packaged preparation 1430414 installed the sealed scene before solver
entry without training. The original resolver misses 32 migrated files; the
reference resolver reads all 64 original dataset records with matching identity
and bytes. These 64 dataset reads are distinct from the 64 generated continuation
samples: four updates × eight logical IDs × two generations.

Packaged continuation 1430440 also passed the exact run specification and hidden
fresh verifier with score 1. The task's `environment/validation.json` records it
separately from native reference run 1430316 and independent audit 1430462.
All seven HF asset files were rehashed at the newly installed model path, and
both final checkpoint payload inventories passed. The fixed evaluation mean
was 0.25; model-quality improvement is not an acceptance condition.
