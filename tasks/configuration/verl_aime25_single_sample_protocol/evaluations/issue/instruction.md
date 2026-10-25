# RL evaluation result diagnosis

Review the staged post-training evaluation and determine whether the observed
AIME25 result is sufficient evidence of a post-training failure, or whether the
result itself is unreliable. Treat the workspace as a historical snapshot:
inspect its source, the evaluation configuration and protocol, the evaluator
code, and the frozen evaluation evidence. Do not assume that a low score is
causal without connecting it to the conditions under which it was measured.

Consider how many questions the benchmark has, how generation is sampled, and
how many samples per question the protocol draws. A single number is not by
itself proof of any single cause.

Do not launch a full training run, use the network, or modify the historical
snapshot. The benchmark, model, and generation settings are fixed.

## Deliverable

Write one concise UTF-8 diagnosis to:

```text
$AUTORLBENCH_RESULTS/agent_answer.md
```

State the verdict, identify the concrete root cause, and explain the causal
chain using specific workspace evidence. Distinguish root causes from symptoms.
Do not propose or implement a fix, and do not write other deliverable files.
