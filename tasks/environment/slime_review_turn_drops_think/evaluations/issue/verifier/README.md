# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold verdict for this task is `unhealthy`; the essential
root cause is that `scripts/feedback_rollout.py` re-renders the multi-turn
message list with the Qwen3 chat template, which strips the first answer's
`<think>...</think>` block, so the review turn is generated without the first
answer's reasoning and turn 1 is trained as a bare answer the policy never
generated (fix: token-in, token-out bookkeeping with trainable generated tokens
and non-trainable observation tokens).

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`; the raw judge
response and the request that produced it go to
`$AUTORLBENCH_RESULTS/judge-raw.json`.
