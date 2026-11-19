# Reward is flat on the repository-repair corpus

*Notes kept by the RL engineer running the repository-repair experiment, after the most recent
grading pass over `corpus/` (pass A).*

## What we see

Three GRPO iterations in a row on this corpus end with the same number: mean reward
`0.0000`, every group degenerate, no advantage, nothing to learn from. `pass_a/summary.json` is
the grading pass behind the last of them: 194 records, 0 rewarded.

That is not what the rollouts look like. Reading them, a good share of the patches in
`corpus/submissions/` are the kind of patch we would merge: they touch the function the bug was
injected into and they undo it. Some are recorded rollouts of the policy; others we staged
ourselves while debugging, including patches written by hand against the injected bug. All of
them come back `unresolved`.

## What we checked

* The repositories are fine. `python3 -m pytest -q` in a fresh copy of any snapshot in
  `corpus/repos/` is green in its own `.venvs/<repo>` environment, in a few seconds.
* The unit tests of the pipeline pass: `python3 -m pytest tests -q` is green.
* This is the corpus the evaluation harness scores checkpoints on, and the evaluation harness
  does not agree with us: it reports a non-zero resolve rate for the same checkpoints whose
  training reward is flat zero.
* Nothing in `pass_a/regrade.jsonl` is marked as an error. Every record was graded and every
  record came back with a reward.

## What is in this directory

```
pass_a/summary.json    the pass's own summary
pass_a/regrade.jsonl   one line per record, as the trainer consumed it
pass_a/raw/*.log       the raw pytest output of 8 records, kept for inspection
```

The sample in `raw/` is the output of the grading command exactly as it ran, for the records
named by the file names.
