# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold verdict for this task is `unhealthy`; the essential
root cause is that 40% of the DAPO-Math-17k answers in `data/pool_clean.jsonl` end
with an invisible U+200B ZERO WIDTH SPACE that `str(r.answer).strip()` in
`scripts/make_rl_data.py` does not remove, so the numeric `match_str` in
`scripts/math_reward.py` (which needs `isnumeric()` targets) scores every sample of
those prompts 0, even correct ones, while the rest of the data scores normally.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`; the raw judge
response and the request that produced it go to
`$AUTORLBENCH_RESULTS/judge-raw.json`.
