# Query pools and the launch gate

This workspace is the data side of an agentic RL run on repository-repair queries. A *pool* is
the set of queries one run trains on; the launcher's gate decides whether a pool may be trained
on, and derives the run's identity and batch configuration from it.

```
pools/<pool_id>/queries.jsonl    one query per line: prompt, tool list, metadata
pools/<pool_id>/manifest.json    what was requested: total, per-source weights and counts
registry/missing_images.json     repo-wide record of instances whose sandbox image is unusable
scripts/build_pool.py            the pool builder: banks + weights -> pool + manifest
scripts/launch_run.sh            the launcher; everything before the final exec is the gate
scripts/consume_pool.py          the trainer stub: reads an accepted pool as a run would
scripts/run_cases.py             entry point: runs the launcher over a case list
cases/cases.jsonl                one launch case per line: a pool and an environment
tests/                           unit tests of the gate
logs/historical/                 one pass of the gate over cases/cases.jsonl, and the notes
                                 it triggered (`incident.md`)
```

## What the gate does

For one launch case, in order:

1. pin `queries.jsonl`, `manifest.json` and the registry by SHA-256,
2. validate the pool: the manifest's own consistency, the rows and their uniqueness, the tool
   list of every query, that no unusable instance is present, and that the manifest's exclusion
   counts match the registry,
3. derive the run identity (`RUN_TAG`) and the pool size the run is given,
4. set the rollout batch configuration,
5. exec the trainer -- here, `consume_pool.py`.

`POOL_VALIDATE_ONLY=1` stops after the checks instead of starting the run.

## Running it

```bash
python3 -B scripts/run_cases.py                       # the whole staged case list, ~1 min
python3 -B scripts/run_cases.py --out /tmp/mine       # same, without touching the harness logs
POOL_DIR=pools/p_balanced POOL_VALIDATE_ONLY=1 \
  POOL_SHA256=... POOL_MANIFEST_SHA256=... POOL_REGISTRY_SHA256=... \
  bash scripts/launch_run.sh                          # one pool, checks only
python3 -m pytest tests -q                            # unit tests of the gate
```

`scripts/run_cases.py` writes `$AUTORLBENCH_RESULTS/experiment_logs/cases.jsonl`, one object per
case:

| field | meaning |
| --- | --- |
| `case_id`, `pool_dir` | which launch case |
| `accepted`, `exit_code`, `message` | what the gate decided and the last line it printed |
| `run` | present when the run started: the run identity, the pool total it was given, the rows it read, its source mix, its batch configuration, and how many prompts repeated inside a rollout |

plus `summary.json`, and per case a directory under `cases/<case_id>/` with the launcher's output
and the run record.

## Rebuilding a pool

```bash
python3 -B scripts/build_pool.py --banks <banks dir> --out pools/mine \
    --total 300 --weights swe_smith=50,swe_rebench=35,r2e_gym=15 \
    --registry registry/missing_images.json --seed 42
```

The builder records the *requested* mix: every source appears in `source_counts`, including one
at weight zero. The per-source banks themselves are not kept in the workspace -- the pools under
`pools/` were built from them during prepare and verified against recorded digests.
