# Reward service for repository-repair RL

This workspace is the reward side of an agentic RL run on repository-repair tasks. A policy is
given a repository with an injected bug and produces a patch; this code decides what that patch
is worth.

```
corpus/manifest.jsonl   one row per instance: repository snapshot, injected bug, required tests
corpus/records.jsonl    one row per graded rollout: which instance, which patch
corpus/patches/         the injected bugs
corpus/submissions/     the rollout patches
corpus/repos/           pristine repository snapshots (git repositories, one commit)
.venvs/<repo>/          the per-repository environment the corpus builder installed
rlgrade/                the grading pipeline
scripts/regrade_run.py  harness entry point: grade every record, write the run log
scripts/run_config.json the pass's knobs (workers, per-record test timeout)
tests/                  unit tests of the pipeline
logs/historical/        the last grading pass over this corpus, and the notes it triggered
                        (`incident.md`)
```

## How one record is graded

`rlgrade.grade_record` stages the record and runs its tests:

1. copy `corpus/repos/<repo>` into a scratch directory,
2. apply the instance's injected bug,
3. apply the rollout's patch,
4. restore the test suite (the rollout is not allowed to decide its own exam),
5. run the test files that hold the instance's required node ids, with the repository's own
   environment,
6. parse the pytest output into per-test statuses,
7. compare them with `fail_to_pass + pass_to_pass` and emit the reward.

## The run log

`scripts/regrade_run.py` writes `$AUTORLBENCH_RESULTS/experiment_logs/regrade.jsonl`, one JSON
object per record, plus `summary.json`. The contract downstream consumers rely on:

| field | meaning |
| --- | --- |
| `record_id`, `instance_id` | which rollout was graded |
| `label` | `resolved`, `unresolved`, `unscorable` or `error` |
| `reward` | `1.0` for `resolved`, `0.0` for `unresolved`, `null` when the record has no reward |
| `failed`, `missing` | required node ids that failed, and that the run never reported |
| `notes` | free-form diagnostics (patch application, exit code, timings) |

`resolved` means *every* required test passed: the tests the bug broke and the tests that were
already green. `unresolved` means the run showed the patch does not do that. `unscorable` is for
records a run could not decide at all -- the trainer drops them instead of feeding a zero into
the advantage. `error` is reserved for the entry point itself failing on a record.

## Running it

```bash
python3 -B scripts/regrade_run.py                 # grade corpus/records.jsonl
python3 -B scripts/regrade_run.py --records my.jsonl --out /tmp/out --workers 8
python3 -m pytest tests -q                        # unit tests
```

Grading is CPU-bound and runs real test suites. The defaults come from
`scripts/run_config.json`: `workers` is the number of records graded concurrently and
`timeout_s` caps one record's test run; both can be overridden with `--workers` and
`--timeout-s`. `--keep-checkouts` leaves each record's scratch checkout in place for
inspection; the raw pytest log of a record is always kept next to it as
`<scratch>/<record_id>/pytest.log`.
