# Issue verifier

The issue tier reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden
`oracle.md` using verdict / root-cause / evidence weights 0.2 / 0.6 / 0.2.
Passing requires verdict = 1, root-cause > 0.8, and evidence > 0.8; the weighted
total alone is not the pass gate. Evidence is scored across the four causal
groups, accepting equivalent visible code/log/reproduction evidence. Exact
corpus counts are illustrative and are not required.
The oracle carries four essential root causes in four different stages of the pipeline, so a
diagnosis that names only the one explaining the flat reward (the missing summary section)
cannot reach the mark. A flat training reward, a corpus of apparently bad rollouts, or a green
unit-test run is not a root cause without the code path that produces it.
