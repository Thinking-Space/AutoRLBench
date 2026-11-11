# Issue verifier

The verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`oracle.md` with the shared LLM judge (`judge.py` and `judge_prompt.md`).

The judge weights verdict, root cause, and evidence agreement at
0.2 / 0.6 / 0.2. A pass requires verdict score 1.0, root-cause score above
0.8, and evidence score above 0.8.

The required verdict is `unhealthy`. A passing diagnosis must connect the
rendered-prompt mutation, `reset_for_retry()` preserving that field, both retry
paths, and the repeated messages-only validation failure. `gold.yaml` is
reviewer-facing only.

The verifier requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. It writes `$AUTORLBENCH_RESULTS/result.json` and the raw judge
exchange to `$AUTORLBENCH_RESULTS/judge-raw.json`.
