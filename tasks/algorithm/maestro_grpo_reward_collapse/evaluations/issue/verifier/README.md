# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. Pass requires
all three conjunctively: the verdict must be exactly the gold verdict, the
root-cause score must exceed 0.8, and the evidence score must exceed 0.8 — a
wrong verdict cannot pass on the other two alone. The gold
verdict for this task is `unhealthy`; the essential root cause is the first-turn
search reconstruction in `VerlToolAgentLoop.run()` in
`verl_tool/agent_loop/verltool_agent_loop.py`. The missing end token interacts
with the subsequent last-token replacement, truncating the closing search tag.
The diagnosis should connect this code path to training reward collapse,
malformed rollout delimiters, and length-exhausted trajectories.

Machine grading reads the hidden `oracle.md`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
