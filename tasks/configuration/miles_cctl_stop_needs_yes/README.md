# cctl job stop silently does nothing without --yes: interactive confirmation aborts non-TTY shell

Managing Qwen3.5-9B arena-rubric RL training jobs on Cybertron. `cctl job stop`
requires the user to type the job name interactively as confirmation. In a
non-TTY shell script, stdin is empty; the prompt reads nothing and aborts with
`ok: false`. All stop commands silently fail — the jobs continue running.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Configuration |
| **Domain** | Arena-Rubric RL |
| **Tags** | `cctl` `job-stop` `interactive-confirmation` `non-TTY` `--yes` `silent-failure` |
| **Framework** | miles (GRPO) |
| **Run time** | >10 minutes of repeated failed stop attempts |
| **Resources** | 0 GPU (cluster management task) |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is a missing `--yes` flag in a cluster job management script.
`cctl job stop tasks/1637987` without `--yes` does the following in a non-TTY
environment (cron job, CI, background shell):

1. Prints: `stop task requires confirmation for "tasks/1637987".`
2. Prints: `Type "tasks/1637987" to continue, or anything else to abort:`
3. Reads stdin — gets nothing (empty pipe or EOF)
4. Prints: `{"error": {"code": "confirmation_required", "message": "stop task aborted; confirmation did not match..."},"ok": false}`
5. Returns exit code 6 (non-zero)

The job is NOT stopped. The script sees `ok: false` but many wrapper scripts
ignore the return value or only check for a 2xx HTTP response.

Over 10+ minutes of repeated `cctl job stop` calls, 7 eval jobs remained
`Running` or `Starting` with no state change despite multiple stop commands.
The only observable symptom was `ok: false` in the cctl JSON output.

The fix is `cctl job stop tasks/$JID --yes`. The `--yes` flag bypasses the
interactive confirmation and returns `{"status": "Killing"}` immediately.

## Reference solution

- **issue**: identify that `cctl job stop` requires interactive confirmation
  (`--yes` or `--confirm <job_id>` flag) and that in a non-TTY shell stdin is
  empty, so the confirmation fails and the job is never stopped; connect this to
  the `ok: false` response and jobs remaining in `Running`/`Starting` state.
  Reject "the jobs are stuck and cannot be stopped" and "the platform has a bug"
  as root causes.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; fix is a one-word flag addition.
