# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` and
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement with weights
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, and
`root_cause_score` and `evidence_score` each greater than 0.8; the weighted
total alone is not a pass condition. The expected verdict is `unhealthy`, and
the canonical root cause is the two-part diagnosis in the oracle: the
single-question 3-10 training set against the varied mostly-0-2
out-of-distribution questions, and the chain from one-question prompts through
the total-count reward to question-ignoring behaviour. Naming only
"distribution shift" or only "over-training" earns partial root-cause credit
(0.3) and cannot pass.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
