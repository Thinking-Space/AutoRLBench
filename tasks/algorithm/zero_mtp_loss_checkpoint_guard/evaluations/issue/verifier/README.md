# Issue verifier

The verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`oracle.md` with the shared LLM judge (`judge.py` and `judge_prompt.md`).

The required verdict is `unhealthy`. A passing diagnosis must identify both
parts of the coupled failure: the one-rollout checkpoint guard rejects a valid
final-only smoke, and the zero-scaled auxiliary MTP loss is returned as absent
but consumed as if it were present. The source of failure is local validation
and loss composition; GPU, network, reward quality, and model capability are
wrong attributions.
