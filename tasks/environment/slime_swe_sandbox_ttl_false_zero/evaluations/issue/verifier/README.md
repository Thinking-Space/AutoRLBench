# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and
`root_cause_score` and `evidence_score` each > 0.8; the weighted total
alone is not a pass condition. The gold verdict for this task is
`unhealthy`; the essential root cause is the three-layer chain: the sandbox
instance TTL (120s) sitting far below the agent time budget (900s) and the
episode-duration tail, the two-layer verify folding (wrapper + blanket
except) that turns every infra failure into a legal 0.0, and the resulting
pollution of the GRPO group baseline (61% of groups, reversed learning
trajectory). Naming only one layer is partial.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. The output is written to
`$AUTORLBENCH_RESULTS/result.json`.
