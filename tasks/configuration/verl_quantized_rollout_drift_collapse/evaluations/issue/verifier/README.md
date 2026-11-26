# Issue tier verification

The judge reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden
`oracle.md`: verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2, pass mark 0.8.
Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`; writes the
result to `$AUTORLBENCH_RESULTS/result.json` and the raw judge response plus the request
that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
