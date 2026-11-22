# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` and
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement with weights
0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, and
`root_cause_score` and `evidence_score` each strictly greater than 0.8; the
weighted total alone is not a pass condition. The expected verdict is
`unhealthy`: the oracle itemizes exactly five independent root-cause items
(validation split leakage at the order-family boundary, non-idempotent retry
after a committed reservation, positive-only progress clipping,
incomplete-receipt full terminal score, and positional optimizer restore) and
six evidence items, each scored as a boolean, so a passing diagnosis must
essentially cover all of them. The clean validation saturation is genuine for
the recorded probe and is not evidence of healthy training transactions; a
diagnosis that attributes the late training failure to a held-out accuracy
crash cannot pass.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
