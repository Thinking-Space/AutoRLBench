# Solution verifier

Two passes decide the score, both running the repositories' real test suites:

1. **the system rerun** of `scripts/regrade_run.py` on the repaired workspace, read from the
   fresh run's `experiment_logs/regrade.jsonl` (the harness clears that directory first), and
2. **a held-out pass** this verifier drives itself: the same entry point and the repaired
   `rlgrade/` package, on a corpus of instances and rollout patches the workspace never
   contained (`resources/hidden/`), graded against the same repository snapshots and
   environments.

Each pass is compared with labels produced by the reference pipeline. A pipeline passes when, on
**both** corpora, it labels no undecided or failing record `resolved`, agrees with the reference
labels on at least 97 % of records, reports a reward consistent with the label it ships
(`1.0` / `0.0` / `null`), and recovers each of `resolved`, `unresolved` and `unscorable` at the
recall floors in `resources/grader_spec.json`.

Before any of that, `integrity` requires the pinned entry point and every corpus file to match
their recorded digests, every repository snapshot to match its content digest with a clean
`git status`, and every pinned environment to still have its packages. Making a record pass by
editing the corpus, a snapshot or an environment fails the task outright.

`resources/` is mounted only in the verifier stage.
