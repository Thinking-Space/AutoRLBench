# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` and
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement with weights
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, and
`root_cause_score` and `evidence_score` each greater than 0.8; the weighted
total alone is not a pass condition. The expected verdict is `unhealthy`: the
run's logged metrics look normal, and the defect is visible only by auditing
the reward path. The canonical root cause is the paired-tag regex in
`extract_non_reasoning_content` becoming a no-op because the chat template
prefills the opening thinking tag into the prompt, so the judge receives the
reasoning block and the answer together. The rollout dumps were deleted
before packaging; the shipped per-step trace and the code-level reproduction
are the citable evidence, and the oracle does not require numbers from
unpublished author-side tests.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
