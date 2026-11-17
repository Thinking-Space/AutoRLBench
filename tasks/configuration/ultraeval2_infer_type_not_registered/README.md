# UltraEval2 infer_type api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1 not registered, all eval submissions fail silently

Evaluating Qwen3.5-9B iter149 on three benchmarks (ahv2, writingbench, tigan).
The eval scheduler spec sets `infer_type=api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1`.
This string is not registered in UltraEval2's inference pool configuration, so every
submission attempt returns `submit_failed: 无可用资源池` at one-minute intervals.
The scheduler job stays Running for 30+ minutes with 75 consecutive failures and no error.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Configuration |
| **Domain** | Arena-Rubric RL |
| **Tags** | `ultraeval2` `infer_type` `无可用资源池` `eval` `think` `Qwen3.5` |
| **Framework** | miles (GRPO) |
| **Run time** | >30 minutes of silent retry (scheduler Running, zero sub-jobs) |
| **Resources** | 0 GPU (eval scheduler task) |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is a single wrong string in the eval scheduler spec:
`infer_type=api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1`

UltraEval2 maps `infer_type` strings to inference cluster configurations. When the
string is not registered, the scheduler cannot find a matching pool and returns
`submit_failed: 无可用资源池` (no available resource pool). The scheduler continues
Running and retries every minute; it never escalates to an error state.

The correct registered string for Qwen3.5-9B think evaluation is:
`api_general_think_sft_t1_tp0.95_rp1.05`

Three observations confirm this:
1. The state file contains 75 `submit_failed` events over 30 minutes
2. A parallel nothink scheduler using `infer_type=api_general_nothink_sft` submits successfully
3. Switching to `api_general_think_sft_t1_tp0.95_rp1.05` immediately produces `submit` events

The incorrect string `tk20_mp0_pp1.5_rp1` looks like a newer version of the
parameter set that had not yet been deployed in UltraEval2 at the time.

## Reference solution

- **issue**: identify that `infer_type=api_general_think_sft_t1_tp0.95_tk20_mp0_pp1.5_rp1`
  is not registered in UltraEval2, connect this to the `submit_failed: 无可用资源池` events
  in the state file, and distinguish it from a pool-capacity issue (which the nothink
  scheduler does not share). Reject "the pool is full" and "the scheduler job is broken".

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; fix is a string change.
