# Incident continuation reference

The task begins with checkpoint3 and already-moved data. The reference resolver
looks up the requested ID and checksum in the current catalog when its cached
manifest path is unavailable. It returns a path; trusted code loads the file,
verifies its bytes, scores the generated answer and trains.

Only loader.py is submitted. There is no reference patch to training, rewards,
checkpoint selection, metrics or fault injection. The system resumes steps4–7
and grades exact logical sample consumption, full native state restoration and
actual updates. Transient reads may be retried; sample replacement fails.

Earlier eight-step from-base records are historical calibration and are not a
claim of restoration from this newly sealed checkpoint. New restore evidence is
recorded separately after verification.

The historical receipts are explicitly separated in `history/`. The current
reference resolver is never copied into the prepared runtime or workspace.
Packaged incident acceptance remains pending a fresh audit; partial restore
observations and synthetic packaging tests do not replace that audit.

For the assessment-first protocol, start report.md with `Verdict: unhealthy`: it describes the received stale resolver even after repair. Explain the current read failure and the chosen change. The paired healthy implementation instead requires `Verdict: healthy` and can be submitted unchanged.
