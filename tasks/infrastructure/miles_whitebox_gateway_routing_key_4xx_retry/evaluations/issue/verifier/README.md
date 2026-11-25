# Issue verifier

`test.sh` runs `judge.py`: an OpenAI-compatible LLM judge compares the agent's
`$AUTORLBENCH_RESULTS/agent_answer.md` against the hidden `oracle.md` using
`judge_prompt.md`, scoring verdict / root cause / evidence (0.2 / 0.6 / 0.2,
pass mark 0.8). `gold.yaml` is the structured record of the same oracle for
tooling; the judge itself reads `oracle.md` only.
