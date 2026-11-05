# Issue verifier

The unchanged official verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden `evaluations/issue/oracle.md` using `judge.py` and `judge_prompt.md`.

The judge reports verdict, root-cause and evidence agreement with weights 0.2, 0.6 and 0.2. Its current pass condition requires `verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`; the weighted total alone does not determine passing.

The expected verdict is unhealthy. The essential attribution is that processed prompt messages cached under a stable name and row count survive a dataset-content revision, causing old messages to be scored against current-row answers. The concrete symbol and evidence requirements are in the hidden oracle, not this verifier code.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The verifier writes `$AUTORLBENCH_RESULTS/result.json`.
