# Miles MiniSWE tool-call parser mismatch

Training Qwen3-30B-A3B with GRPO on 16 SWE-bench Verified prompts through Miles, Harbor, and mini-swe-agent configures SGLang with the `qwen3_coder` tool-call parser instead of the checkpoint's required `qwen25` parser, so tool calls remain literal assistant text, every TITO-v2 rollout reports an assistant-text mismatch, and 64 logical rollouts become 192 training leaves.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Configuration |
| **Domain** | Agentic RL |
| **Tags** | `tool-call-parser` `session-mismatch` `rollout-fanout` `issue-only` |
| **Framework** | Miles (GRPO), commit `64772f50b20b6e39051922a4033f3cb6019985dc`; Harbor commit `0c2e1953c638f392ebbcf31d04883cb5e16edb22` |
| **Run time** | ~20 minutes |
| **Resources** | 8 × H100 (80 GB) |

## Difficulty

The defect is a one-line launch configuration in `workspace/run_miles.py`: the
Qwen3-30B-A3B checkpoint is served with
`--sglang-tool-call-parser qwen3_coder`, while this checkpoint requires the
`qwen25` tool-call parser. The wrong parser does not reject the request or fail
during startup. Instead, SGLang leaves the model's tool-call JSON in the
assistant text instead of separating it as structured tool-call content.

The strict TITO-v2 message matcher then compares that literal assistant text
with the session server's expected path and records `assistant_text`
mismatches. The frozen rollout records contain 187 such mismatches: the
`actual_text` includes a trailing object such as
`{"name": ..., "arguments": ...}` after the reasoning text, while the
`expected_text` stops before that tool-call object. All 64 logical rollouts
are affected.

The accounting impact is uniform and severe:

| metric | value |
| --- | --- |
| logical rollouts | 64 |
| training leaves | 192 |
| leaves per rollout | 3 |
| extra training leaves | 128 |
| sample inflation ratio | 3.0 |
| fan-out rollouts | 64 |
| mismatch rollouts | 64 |
| TITO session mismatch rate | 1.0 |

Harbor still creates only 64 trials, so the extra leaves are not additional
MiniSWE episodes. The run records 72 `AgentError` and 120 `Submitted` exit
statuses, mean reward 0.0, and mean total time 639.228 seconds. The trainer
update exits nonzero after the accounting evidence is collected, but the
configuration error is already visible in the mismatch records.

A one-line `qwen25` control run reduces training leaves from 192 to 164,
sample inflation from 3.0× to 2.562×, fan-out rollouts from 64 to 30, and
mismatch rollouts from 64 to 56. That control establishes the parser setting as
a real configuration defect. The remaining fan-out is a separate MiniSWE
integration issue and is intentionally not this task's canonical answer.

## Reference solution

- **issue**: identify the `qwen3_coder` tool-call parser as the wrong parser for
  this Qwen3-30B-A3B checkpoint, state that the required parser is `qwen25`,
  and connect it to literal tool-call JSON in assistant text, strict
  TITO-v2 `assistant_text` mismatch, and the uniform three-leaf expansion from
  64 logical rollouts to 192 training samples. Attributing this task's uniform
  inflation to MiniSWE itself, the TITO matcher, Harbor, the `qwen3` reasoning
  parser, AGS, or SWE-bench data fails.

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`,
  `root_cause_score > 0.8`, and `evidence_score > 0.8`. No training, sandbox,
  network, or model access is required.
