# Training entry fatals immediately when judge URL file is absent instead of waiting

Training Qwen3.5-9B with arena-rubric GRPO using miles. The training entry script
checks for the judge URL file at startup with a one-shot conditional and exits
immediately if absent. The judge job was still `Queued` and had not written its
URL file, so training fataled within seconds of starting.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Configuration |
| **Domain** | Arena-Rubric RL |
| **Tags** | `judge-url` `FATAL` `startup` `wait-loop` `miles` `arena-rubric` `Qwen3.5` |
| **Framework** | miles (GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | < 5 seconds (crash at startup) |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is a one-shot conditional in the training entry script's startup logic:

```bash
_JUDGE_URL_FILE=$(ls /scratch/judge_ips/*/*.url 2>/dev/null | head -1)
if [ -z "${_JUDGE_URL_FILE}" ] || [ ! -f "${_JUDGE_URL_FILE}" ]; then
  echo "[FATAL] 找不到判官 URL 文件" >&2
  exit 1
fi
```

The judge job was submitted concurrently with the training job. The judge runs on
a separate pool (rl2 HIGH) and was still `Queued` when training started. No URL
file existed yet under `scratch/judge_ips/`. The check found nothing and exited
immediately with code 1.

The training job completed its container initialization, reached the startup check,
found no URL file, and exited. The entire failure takes under 5 seconds. The job
shows `容器 pytorch 失败（Error），退出码 1` which looks like a framework crash —
the actual cause is a deliberate `exit 1` in the shell script.

The fix is a polling loop that waits up to 90 minutes (180 × 30s) for the URL file
to appear and verifies it passes a health check before proceeding.

## Reference solution

- **issue**: identify the one-shot `if` check in the entry script's judge-URL logic,
  that the judge was still Queued when training started, and connect these to the
  `[FATAL] 找不到判官 URL 文件` log line and the `exit 1`. Reject "the judge failed"
  (judge eventually ran) and "infra failure" (the exit was deliberate).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; fix is replacing the one-shot check with a
polling loop.
