# Issue verifier

The verifier compares `$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden
`oracle.md` using the configured OpenAI-compatible judge model.

The weighted score uses verdict/root-cause/evidence weights 0.2/0.6/0.2.
The standard judge implementation passes only when verdict_score is exactly
1.0 and root_cause_score and evidence_score are both strictly above 0.8.

The expected verdict is unhealthy. The cause is user-side cleaning in
`Tau2Runner._clean_messages`, reached through `_make_completion_fn`, erasing
the simulator's own tool calls and results after tau2 perspective conversion.
The canonical details and recorded observations are in `oracle.md`.

Authentication uses `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. `AUTORLBENCH_JUDGE_TIMEOUT_S` controls the request timeout
and defaults to 60 seconds in the standard scaffold. This provider's tests
set it to 300 seconds. The result is written to
`$AUTORLBENCH_RESULTS/result.json`.
