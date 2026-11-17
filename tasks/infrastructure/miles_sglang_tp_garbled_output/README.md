# SGLang TP>1 produces garbled HTTP 200 rollout output for Qwen3.5 (issue 21039)

Training Qwen3.5-9B with arena-rubric GRPO using miles. `ROLLOUT_NUM_GPUS_PER_ENGINE=2`
causes sglang 0.5.9 to run with TP=2 for Qwen3.5. Due to upstream sglang issue 21039,
Qwen3.5 produces garbled HTTP 200 responses at TP>1, so all rollouts are nonsense
and the rubric reward stays near 0 throughout training.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Infrastructure |
| **Domain** | Arena-Rubric RL |
| **Tags** | `sglang` `tensor-parallel` `Qwen3.5` `garbled-output` `TP>1` `issue-21039` |
| **Framework** | miles (GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | Full training completes; defect spans all training steps |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is `ROLLOUT_NUM_GPUS_PER_ENGINE=2` in the wrapper script. This tells
miles to start the sglang rollout engine with TP=2 (two GPUs per engine instance).
Due to upstream sglang issue 21039, Qwen3.5 produces garbled responses when served
with TP>1 in sglang 0.5.9.

The garbled responses look completely normal to the HTTP client:
- HTTP 200 status
- Valid JSON structure with `choices[0].message.content`
- Normal token counts and `finish_reason: stop`
- No server-side errors or timeouts

The response content is semantically nonsense — typically a mixture of random CJK
characters, repeated tokens, or incoherent fragments. The rubric judge receives
these responses and scores them near 0 (there is no meaningful content to judge
against the rubric criteria). Training reward stays flat near 0 throughout all
200 rollouts.

The launch config log shows `ROLLOUT_NUM_GPUS_PER_ENGINE 2`, which appears fine.
The rollout throughput looks normal. The only clear symptoms are:
- `arena_rubric/score_mean` near 0 from step 0 through end
- CJK ratio in rollout responses near 0 (for a Chinese question dataset)
- Repeated n-grams in rollout content

The fix is `ROLLOUT_NUM_GPUS_PER_ENGINE=1`, which starts one sglang engine per
GPU and avoids TP>1.

## Reference solution

- **issue**: identify that `ROLLOUT_NUM_GPUS_PER_ENGINE=2` causes sglang 0.5.9 to
  run with TP=2 for Qwen3.5, that sglang issue 21039 makes Qwen3.5 produce garbled
  HTTP 200 responses at TP>1, and connect this to near-zero reward throughout
  training. Reject "the rubric judge failed", "the model did not converge", and
  "the checkpoint was not loaded" as root causes.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; the fix requires restarting training with
a corrected launch script.
