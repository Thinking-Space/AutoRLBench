# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold verdict for this task is `unhealthy`; the essential
root cause is the fp8 rollout KV cache (`run_grpo.sh`) combined with verl's
level-2 vLLM sleep and weight reload, which leaves vLLM 0.11's
`Attention._k_scale` / `_v_scale` at 0.0, so every rollout degenerates to `!`.
A generic "fp8 is imprecise without calibration" explanation earns only partial
root-cause credit.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`; the raw judge
response and the request that produced it go to
`$AUTORLBENCH_RESULTS/judge-raw.json`.
