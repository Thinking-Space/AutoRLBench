# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and
`evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold
verdict for this task is `unhealthy`; the essential root cause is the interaction between the KL
penalty applied in the reward (`algorithm.use_kl_in_reward=True`, coefficient 0.001) and GRPO's
per-group division by the standard deviation (`norm_adv_by_std_in_grpo=True`, verl's default): in
prompt groups whose answers are all right or all wrong the penalty is the only variation and is
scaled to a unit-scale advantage that pulls the policy back toward the initial model.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`, and the raw judge
response plus the request that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
