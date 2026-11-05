# Issue verifier

The LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden
`evaluations/issue/oracle.md` (`judge.py` + `judge_prompt.md`). Verdict,
root-cause and evidence agreement are weighted 0.2 / 0.6 / 0.2; passing
requires verdict 1.0 with both the root-cause and evidence scores strictly
above 0.8. The essential attribution is the return-accumulator reset at
masked tool observations in
`vendor/verl/verl/trainer/ppo/core_algos.py::compute_reinforce_plus_plus_outcome_advantage`,
which clears the terminal reward before the backward scan reaches earlier
assistant action tokens.

`gold.yaml` is a reviewer-facing reference; machine grading reads `oracle.md`,
not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
