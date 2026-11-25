# Solution verifier

`test.sh` runs `tests/grade_submission.py`, which gates a repaired workspace on:

1. **sealed-file integrity** — the routing proxy, rollout plugin, recipe,
   experiment driver, dataset, and recorded evidence must hash-match the
   pristine shipped copies;
2. **the fresh rerun's summary** — `$AUTORLBENCH_RESULTS/experiment_logs/summary.json`
   from the harness-driven `bash scripts/run_experiment.sh`: >= 5 of 10 rollout
   iterations completed, zero routing-key rejections, zero gateway inner retry
   lines;
3. **the retry policy** — the workspace's `miles.utils.http_utils._post` must
   fail fast on 400/401/403/404/422 (one attempt, then raise) and still retry
   408/429/500/503 and transport errors, checked directly with a scripted
   client.

The result lands in `$AUTORLBENCH_RESULTS/result.json`; exit code 0 only on a
pass.
