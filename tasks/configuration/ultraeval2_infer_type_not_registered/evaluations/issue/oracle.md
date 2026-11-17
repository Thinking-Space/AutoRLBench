# Canonical issue oracle

Reconciled from session-documented evidence (Qwen3.5-9B arena-rubric RL, 2026-11-16;
evidence is a faithful synthetic reconstruction from documented framework behavior).

Expected verdict: **unhealthy**.

## Root cause

- `scripts/submit_eval.sh::INFER_TYPE` parameter
  - kind: `hparam`
  - observed: the eval scheduler spec uses
    `infer_type=api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1`.
    This string is not registered in UltraEval2's inference pool configuration,
    so the scheduler cannot map it to any available inference cluster. Every
    submission attempt returns:
    ```
    submit_failed: 无可用资源池
    ```
    at approximately one-minute intervals. The scheduler job remains `Running`
    with zero successful sub-job submissions and no error or warning in the job
    log beyond the per-attempt `submit_failed` entries in the state file.
    Over 30 minutes, 75 consecutive `submit_failed` events were recorded.
  - expected: `infer_type=api_general_think_sft_t1_tp0.95_rp1.05` is registered
    in UltraEval2 and resolves to an available inference cluster; submissions
    succeed and produce `submit` events in the state file.

## Observable impact

- State file log contains only `submit_failed` events (75 in 30 minutes).
- Scheduler job status is `Running` throughout — it appears healthy from the
  job list but produces no inference sub-jobs.
- No inference jobs (ahv2, writingbench, tigan) appear under the user's job list.
- Switching `infer_type` to `api_general_think_sft_t1_tp0.95_rp1.05` immediately
  produces `submit` events.

## Trigger conditions

- `infer_type` string not registered in the UltraEval2 pool configuration.
- The scheduler treats an unknown `infer_type` as "no matching pool" rather than
  an error, so the job stays Running and retries silently.

## Documentation reconciliation

- "The inference pool is full" is a wrong attribution: the same scheduler with
  `infer_type=api_general_nothink_sft` submits successfully in the same time window.
- "The scheduler job is broken" is a wrong attribution: the job is Running
  normally; only the `infer_type` lookup fails.
- "The eval spec is otherwise correct" is true; the only defective field is
  `infer_type`.
