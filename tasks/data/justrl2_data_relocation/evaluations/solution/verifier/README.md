# Structural integrity grading

`test.sh` reads only the fresh result directory declared by `run_spec.json`.
The system performs the run after the agent submits `report.md`. All eight
updates and the midpoint migration remain mandatory.

The grade is binary `structural_coverage`. Passing requires the intended sample
IDs and original question/answer bytes, a real moved hard directory, no missing
data after migration, one-half hard samples in every admitted batch, complete
read/terminal/admission counts, successful actor and critic optimizer updates,
all structured fixed evaluations and final native model/optimizer payloads.
The fresh grader rehashes these payloads. No sampled or fabricated summary flag
substitutes for the underlying records.

Calibration replay: one stale-mapping run scores 0; one repaired run scores 1;
the no-migration control scores 0 because it omits the required event. The final
hard-share margin is 0 versus 0.5, checked at every post-event step. This is a
data-contract repair criterion, not a threshold derived from model accuracy.
Initial good coverage cannot pass without all eight steps and final artifacts.

The unit tests extract the pinned public audit archive bundled in the companion
proposal. They do not need private local paths. Explicit `--home` calibration
allows archives without tensor payloads and reports that limitation; normal
harness grading requires the actual payload files.
