# MiniSWE format-error retries inflate Miles TITO training leaves

This issue-only task diagnoses a Miles GRPO run with Qwen3-30B-A3B on 16
SWE-bench Verified prompts in which MiniSWE format-error retries make 64
intended logical rollouts yield 164 TITO-v2 training leaves.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Environment |
| **Domain** | SWE |
| **Tags** | `format-error` `rollout-fanout` `tito-tree` `issue-only` |
| **Framework** | Miles GRPO (Miles 64772f50, Harbor 0c2e1953, MiniSWE 2.4.6) |
| **Run time** | ~24 minutes |
| **Resources** | 8 × H100, 64 logical rollouts, 16 SWE-bench Verified prompts |

## Difficulty

The task requires connecting a user-facing MiniSWE format error to TITO-v2 tree
attachment semantics and then to training-leaf accounting. It also requires
distinguishing the residual `assistant_text` token mismatch from the fan-out
mechanism.

## Reference solution

- **issue**: MiniSWE omits parse-failed assistant attempts from subsequent
  request history. TITO records each attempt as a sibling node, and the
  same-prompt retry picker cannot drop siblings whose prompt contains a new
  format-error user message. The frozen run consequently yields 164 training
  samples for 64 logical rollouts.

## Verification

- **issue**: The judge checks that the report states an unhealthy verdict, the
  MiniSWE format-error non-replay mechanism, TITO sibling attachment, picker
  behavior, and the exact frozen metrics. It rejects parser, sandbox, task
  selection, or model-capacity explanations as the primary root cause.
  This is issue-only because the upstream MiniSWE repair would change
  request-history semantics and no graded control rerun is available.
