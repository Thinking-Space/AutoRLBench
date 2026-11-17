# Canonical issue oracle

Reconciled from session-documented evidence (2026-11-16/17 Cybertron cluster;
evidence is a faithful synthetic reconstruction from documented behavior).

Expected verdict: **unhealthy**.

## Root cause

- `scripts/run_training.sh` — judge URL detection logic
  - kind: `script`
  - observed: the training entry script checks for the judge URL file at startup
    with a one-shot conditional:
    ```bash
    _JUDGE_URL_FILE=$(ls /scratch/judge_ips/*/*.url 2>/dev/null | head -1)
    if [ -z "${_JUDGE_URL_FILE}" ] || [ ! -f "${_JUDGE_URL_FILE}" ]; then
      echo "[FATAL] 找不到判官 URL 文件" >&2
      exit 1
    fi
    ```
    The judge job was still `Queued` at training start and had not yet written its
    URL file. The check immediately found no file and exited with code 1. No retry
    or wait was attempted.
  - expected: the startup logic polls `scratch/judge_ips/*/*.url` in a loop (e.g.
    every 30 seconds for up to 90 minutes), waits for the judge to become available,
    and only fatals if the timeout is exceeded. This way training can start before
    the judge is Running and still proceed once it comes up.

## Observable impact

- The training job exits with code 1 within seconds of starting.
- The driver log contains `[FATAL] 找不到判官 URL 文件` and no rollout output.
- `cctl job get` shows `errorMessage: 容器 pytorch 失败（Error），退出码 1`.
- The judge job eventually becomes Running and writes its URL file, but training
  has already exited by then.

## Trigger conditions

- The training job starts before the judge job has transitioned from `Queued`
  to `Running` and written its URL file to `scratch/judge_ips/`.
- The startup check is a one-shot conditional with no retry or wait.

## Documentation reconciliation

- "The judge failed" is a wrong attribution: the judge eventually ran successfully;
  the defect is that training did not wait for it.
- "The job had an infra failure" is a wrong attribution: the exit code 1 came from
  an explicit `exit 1` in the entry script, not from a container or hardware issue.
- "Training should run after the judge" is a downstream operational note, not the
  root cause; the root cause is the lack of a wait loop in the entry script.
