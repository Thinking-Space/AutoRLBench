# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

MiniSWE 2.4.6's format-error control flow does not replay a sampled assistant
attempt that failed tool-call parsing.

- `workspace/mini_swe_agent/src/minisweagent/models/litellm_model.py` calls
  `_parse_actions(response)` before returning the assistant message. A missing,
  malformed, unknown, or truncated tool call raises `FormatError` from inside
  `LitellmModel.query`.
- `DefaultAgent.query` therefore never appends that assistant response to
  `self.messages`; `DefaultAgent.run` catches `FormatError` and appends only
  the user-side correction message returned in `e.messages`.
- On the next request MiniSWE sends the old prefix plus that new user message,
  but omits the already-sampled assistant attempt. TITO has nevertheless
  committed the attempt as a tree node. `find_attach_point` consequently
  attaches the retry below the earlier parent, and the retry becomes a sibling
  rather than an extension of the failed assistant node.
- `drop_same_prompt_retries` cannot remove these siblings: its
  `sibling_prompt_must_match=True` rule sees the appended format-error user
  message, so the later sibling has a different prompt. Every surviving sibling
  path becomes a training leaf.

Naming MiniSWE's non-replay of parse-failed assistant attempts, TITO's sibling
attachment, and the picker's inability to classify those siblings as identical
retries is a complete attribution. A diagnosis that merely reports leaf
inflation, blames MiniSWE generically, or attributes fan-out to the strict
matcher, Tencent AGS, Harbor, SWE-bench difficulty, or model quality earns at
most partial root-cause credit.

## Observable impact

The authoritative values from
`workspace/observed_run/experiment_logs/metrics.json` are:

- 64 intended logical rollouts and 164 training samples;
- 2.562 sample-inflation ratio and 100 extra training samples;
- 30 fan-out rollouts, with a maximum of 21 leaves per logical rollout;
- 56 mismatch rollouts and a TITO session mismatch rate of 0.875;
- sample statuses of 128 completed and 36 truncated;
- 64 Harbor trials, with 97 `AgentError` and 67 `Submitted` exits;
- mean reward 0.0.


## Scope constraint

The launcher uses the required `qwen25` tool-call parser and `qwen3`
reasoning parser. Residual `assistant_text` mismatches are a separate
template/tokenization signal and are not the canonical fan-out cause. Correct
diagnoses may mention that signal, but must not present it as the primary
mechanism for the 164-leaf accounting result.
