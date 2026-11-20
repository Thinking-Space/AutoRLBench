# Solution verifier

Two passes decide the score:

1. **the system rerun** of `scripts/run_cases.py` on the repaired workspace, read from the fresh
   run's `experiment_logs/cases.jsonl`, and
2. **a held-out pass** this verifier drives itself: the same entry point and the repaired
   launcher, over a case list the workspace never contained (`resources/hidden_cases.jsonl`),
   against the pools the workspace already has.

Every case is turned into an outcome -- `accept`, `reject` or `unsafe-accept` -- from the run's
own record and the pool on disk, never from what the gate printed about itself, and compared with
the reference outcome. A gate passes when, on **both** lists, it reproduces every reference
outcome, never accepts unsafely, never accepts a case the reference rejects, and never rejects a
pool the builder legitimately produces.

Before any of that, `integrity` requires the pinned entry point, the stub consumer, the pool
builder, the staged case list and the registry to match their recorded digests, and every pool to
match its recorded digests file by file. Making a case pass by editing a pool, a manifest or the
registry fails the task outright.

`resources/` is mounted only in the verifier stage.
