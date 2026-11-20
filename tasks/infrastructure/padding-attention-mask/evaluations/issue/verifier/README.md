# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` and
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement with weights
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, and
`root_cause_score` and `evidence_score` each greater than 0.8; the weighted
total alone is not a pass condition. The expected verdict is `unhealthy`, and
the canonical root cause is the missing `ALL_MASK_ATTENTION_FUNCTIONS`
registration for the TurboQuant custom attention, the resulting `None` mask
forwarded to SDPA, and the layout-dependent actor scoring it causes. Exact
numeric values from unpublished author-side tests are not required; the
oracle names the shipped evidence a diagnosis may rely on. An accurately
supported diagnosis of the separate native-versus-HF prefill discrepancy
earns additional credit but must stay distinct from the target defect.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
