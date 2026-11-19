# Issue verifier

The issue tier reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden
`oracle.md` using verdict / root-cause / evidence weights 0.2 / 0.6 / 0.2. The pass mark is 0.8.
The oracle carries four essential root causes in four different stages of the pipeline, so a
diagnosis that names only the one explaining the flat reward (the missing summary section)
cannot reach the mark. A flat training reward, a corpus of apparently bad rollouts, or a green
unit-test run is not a root cause without the code path that produces it.
