# Issue verifier

`test.sh` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md` with an LLM judge. It does not read `submission.json`.

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold
verdict for this task is `unhealthy`; the root cause is the observation
loss mask in `generate_with_search.py::generate`.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
