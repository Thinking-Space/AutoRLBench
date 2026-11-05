# RL training run diagnosis

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a historical snapshot: inspect its source,
effective configuration, launch scripts, data processing, reward logic, logs,
tracked metrics, and recorded rollout trajectories. Do not assume that an unusual
setting or implementation is causal without connecting it to observed behavior.

Base the diagnosis on training reward anomalies and rollout trajectories,
including action formatting, tool interactions, termination, and truncation.
Connect these observations to the source code and effective configuration.
A reward drop or truncated log alone is not proof of a particular root cause.

Do not launch a full training run, use the network, or modify the historical
snapshot. The dataset, held-out set, model, training-step budget, and evaluation
interval are fixed.

## Deliverable

Write one concise UTF-8 diagnosis to:

```text
$AUTORLBENCH_RESULTS/agent_answer.md
```

State the verdict, identify the concrete root cause or causes, and explain the
causal chain using specific workspace evidence. Distinguish root causes from
symptoms. Do not propose or implement a fix, and do not write other deliverable
files.
