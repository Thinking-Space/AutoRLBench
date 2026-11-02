# Preserved checkpoint continuation

`build.py` performs one-time checkpoint authoring from the pinned public assets.
It runs steps 0–3 under the original eight-step schedule, then seals actor/critic
model, Adam, scheduler, RNG and data-cursor state. `scene.py` relocates difficult
payloads in an isolated copy while retaining their original manifest paths.

Task preparation installs that exact snapshot. It does not train. `run.py`
restores the saved state and runs steps 4–7; `verify_resume.py` audits the resulting
sample identities, optimizer updates, checkpoints and fixed evaluation.

`launch.py` starts through isolated Python flags. `resolve_worker.py` interprets
a bounded read-only path-query subset rather than executing submitted Python.
Only the requested relative path crosses into the trusted reader, which checks
the original bytes and logical ID independently.

`evidence/` contains the historical real GPU continuation and its independent
acceptance. The current security boundary was added later and is covered by
CPU regression tests; those receipts are not relabeled as a new GPU validation.
The full 66.7 GB snapshot is delivered as a pinned local directory, not through Git.
