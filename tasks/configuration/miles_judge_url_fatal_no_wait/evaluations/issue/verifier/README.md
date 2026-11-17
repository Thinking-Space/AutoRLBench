# Issue tier verification

An LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden
`oracle.md` on three axes: verdict / root cause / evidence, weighted
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0` with `root_cause_score`
and `evidence_score` each above 0.8.

The judge requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY` in the environment. It writes the result to
`$AUTORLBENCH_RESULTS/result.json` and the raw judge response plus the request
that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
