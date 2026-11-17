# Canonical issue oracle

Reconciled from session-documented evidence (2026-11-16/17 Cybertron cluster;
evidence is a faithful synthetic reconstruction from documented behavior).

Expected verdict: **unhealthy**.

## Root cause

- `scripts/stop_jobs.sh` — `cctl job stop` invoked without `--yes`
  - kind: `script`
  - observed: `tools/cctl.sh job stop tasks/1637987` is called in a non-TTY
    shell script. `cctl` requires the user to type the job name interactively
    as confirmation:
    ```
    stop task requires confirmation for "tasks/1637987".
    Type "tasks/1637987" to continue, or anything else to abort:
    ```
    In a non-TTY environment, stdin is empty (EOF). `cctl` reads nothing, treats
    it as a non-matching input, and returns:
    ```json
    {"error":{"code":"confirmation_required",
     "message":"stop task aborted; confirmation did not match \"tasks/1637987\"",
     "suggestion":"Re-run with --yes or --confirm tasks/1637987"},
     "ok":false}
    ```
    The job is NOT stopped; exit code is 6. This happened for all 7 stop calls
    over 10+ minutes; all jobs remained in `Running` or `Starting` state.
  - expected: `tools/cctl.sh job stop tasks/1637987 --yes` bypasses the
    confirmation prompt and returns `{"status":"Killing"}` immediately.

## Observable impact

- `cctl job stop` returns `ok: false` and `code: confirmation_required` for
  every invocation.
- Jobs remain in `Running` or `Starting` state after multiple stop attempts.
- The script exits with code 6 (non-zero) but continues the loop.

## Trigger conditions

- `cctl job stop` called in a non-TTY environment (cron job, CI, background
  shell script) without the `--yes` or `--confirm <job_id>` flag.

## Documentation reconciliation

- "The jobs are stuck and cannot be stopped" is a wrong attribution: the jobs
  respond to stop when `--yes` is used; they remain running only because the
  confirmation was never submitted.
- "The platform has a bug with these job IDs" is a wrong attribution: the same
  `cctl job stop tasks/$JID --yes` immediately returns `Killing`.
- The exit code 6 is a signal that confirmation was required but not given;
  it does not indicate a network or authentication failure.
