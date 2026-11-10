# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold
verdict for this task is `unhealthy`; the essential root cause is the IcePop
token-mask window narrowed to [0.67, 1.5] by
`actor.rejection_sampling.lower` / `actor.rejection_sampling.upper` in `run.sh`
(the shipped `examples/math/gsm8k_icepop.yaml` window is [0.5, 5]).

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`, and the raw judge
response plus the request that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
