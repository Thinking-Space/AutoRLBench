# Data contract

Entry point:`../test.sh`. The system runs task-root `run_spec.json` first.
The grader reads only this attempt's `experiment_logs`,the submitted workspace,
and immutable assets delivered by prepare. Every check listed in `grader_spec.json`
must pass for binary score1. No scalar training metric or learning curve is scored.
