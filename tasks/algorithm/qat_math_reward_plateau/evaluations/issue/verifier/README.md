# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` and
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement with weights
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, and
`root_cause_score` and `evidence_score` each strictly greater than 0.8; the
weighted total alone is not a pass condition. The expected verdict is
`unhealthy`: the run updates but its training objective is defective. The
rubric awards 0.5 for each of the two root-cause mechanisms (the on_policy
single-minibatch branch substituting a detached copy of the actor's current
scores for the rollout log probabilities, and the Minerva `Answer:` extractor
rejecting valid boxed answers), so a passing diagnosis must correctly trace
both; the evidence score is one third per evidence group (PPO, reward,
training behavior). Author-only controls, a specific boxed example, and the
inherited block-alignment probe are not required.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
