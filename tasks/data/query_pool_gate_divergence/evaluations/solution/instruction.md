# Repair the query pool launch gate

The workspace holds the data side of an agentic RL setup:

```
pools/<pool_id>/{queries.jsonl,manifest.json}  the query pools, built by scripts/build_pool.py
registry/missing_images.json                   repo-wide record of unusable instances
scripts/launch_run.sh                          the launcher and its pool gate
scripts/build_pool.py                          the pool builder
scripts/consume_pool.py                        stub trainer: reads an accepted pool as a run would
scripts/run_cases.py                           entry point: runs the launcher over a case list
cases/cases.jsonl                              one launch case per line: a pool and an environment
tests/                                         unit tests of the gate
logs/historical/                               one pass of the gate, and the notes it triggered
```

Decide whether the gate's decisions mean what they claim to mean. If they do not, find the root
cause or causes and repair them in the staged workspace at `/workspace`. Do not assume a check is
at fault without connecting it to observed behaviour, and do not stop at the first contributing
factor you find.

**Do not use the network.** Everything needed is staged locally. You may run the launcher and the
builder while you work: `python3 -B scripts/run_cases.py --out /tmp/<yours>` runs the whole case
list without touching the harness log directory, and takes a couple of seconds.

## What the gate is for

A launch case asks for one experiment. The gate's job is to let that experiment start only if it
is the experiment the case asked for, and to refuse otherwise — loudly, before any training
begins.

Two consequences the system holds the gate to:

* **Whatever the gate starts must be the pool it validated.** If a run begins, its identity, the
  pool size it is given, the data it reads and the batch configuration it runs under all have to
  describe the pool on disk. A run that starts on something else is worse than a refused one: the
  refusal is visible and this is not.
* **Whatever the pool builder legitimately produces must be launchable.** A gate that refuses
  pools this repository's own tooling writes is not strict, it is broken, and the experiments it
  blocks are real experiments.

A launch case carries more than a pool path. Everything it states is part of the experiment it
is asking for, including a run tag it pins by hand.

The pool, its manifest and the registry are inputs, not drafts. One that is internally
inconsistent or malformed is a reason to refuse the launch and say so, never to repair it on the
fly and continue.

## How this is graded

The system reruns `python3 -B scripts/run_cases.py` and reads the `experiment_logs/cases.jsonl`
it writes. Each case is then judged from the run's own record and the pool on disk -- never from
what the gate printed about itself -- as one of: the gate refused; the gate started a run that
matches the pool; or the gate started a run that does not. The same judgement is then applied to
launch cases that are not in `cases/cases.jsonl`.

## What you may change

`scripts/launch_run.sh` and `tests/`.

These are compared against pristine copies and must not be modified: `scripts/run_cases.py`,
`scripts/consume_pool.py`, `scripts/build_pool.py`, `cases/cases.jsonl`, `registry/` and
everything under `pools/`. The pools are the data the gate exists to judge; editing a pool, a
manifest or the registry to make a case pass is not a repair of the gate, and a modified copy
fails grading outright.

When you are done, submit `/workspace/report.md`: the verdict and the evidence that decides it,
each root cause with the evidence connecting it to the observed behaviour, every file you changed
and what you changed in it, and which cases your repair re-decides and in which direction. A
missing or empty report scores 0.
