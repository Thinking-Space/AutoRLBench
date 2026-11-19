# Reward pipeline diagnosis

The workspace holds the reward side of an agentic RL run on repository-repair tasks: the
grading pipeline, the rollout corpus it grades (repository snapshots, injected bugs, required
tests, recorded rollout patches), the per-repository environments, and the logs of an earlier
grading pass over that corpus.

Decide whether this reward pipeline reports what it claims to report. If it does not, find the
concrete root cause or causes in the code and the corpus data, and explain the causal chain
from each of them to what the logs show, with specific file, record and number evidence.
Separate root causes from symptoms and from unrelated noise, and say which rollouts each cause
mislabels and in which direction.

You may run the pipeline and the repository test suites locally; they are small and the
environments are installed. Do not use the network, and do not modify the archived logs.

Write one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. Do not propose a
fix.
