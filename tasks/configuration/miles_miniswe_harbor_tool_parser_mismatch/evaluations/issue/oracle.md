# Canonical root-cause oracle

Expected verdict: **unhealthy**.

## Root cause

`workspace/run_miles.py` configures the Qwen3-30B-A3B SGLang rollout engine
with `--sglang-tool-call-parser qwen3_coder`. This checkpoint requires the
`qwen25` tool-call parser. The wrong parser does not separate the model's
tool-call object from assistant text. That literal tool-call JSON remains in
`actual_text`, the strict TITO-v2 message matcher compares it against the
different expected session path, and every logical rollout is marked with an
`assistant_text` mismatch. The session tree consequently yields three training
leaves per intended logical rollout.

## Observable impact

The authoritative values from
`workspace/observed_run/experiment_logs/metrics.json` are:

- 64 intended logical rollouts;
- 192 training samples;
- 3.0 sample-inflation ratio;
- 128 extra training samples;
- 64 fan-out rollouts;
- 64 mismatch rollouts and a TITO session mismatch rate of 1.0;
- 64 Harbor trials, with 72 `AgentError` and 120 `Submitted` exit statuses;
- mean reward 0.0 and mean total time 639.228 seconds.

The rollout records contain 187 `assistant_text` mismatch entries. In these
entries, `actual_text` contains a trailing literal tool-call object such as
`{"name": ..., "arguments": ...}` after the assistant reasoning text, while
`expected_text` ends before that object. The literal tool-call text is the
direct signal that the configured parser did not decode the tool call.

The wrong attribution is MiniSWE itself, Harbor, the TITO-v2 strict matcher,
the `qwen3` reasoning parser, the Tencent AGS adapter, or the SWE-bench task
selection. This task's canonical root cause is only the `qwen3_coder` versus
`qwen25` tool-call-parser configuration mismatch.

## Scope constraint

Correcting the parser reduces, but does not eliminate, fan-out. The remaining
post-`qwen25` behavior is a separate issue and is not the canonical answer for
this task. A diagnosis must not claim that MiniSWE has no independent problem;
it must only avoid attributing this task's uniform three-leaf expansion and
all-rollout assistant-text mismatch to that separate issue.
