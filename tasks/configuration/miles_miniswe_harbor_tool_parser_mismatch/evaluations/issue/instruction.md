# Agentic RL tool-call parser diagnosis

Review the staged public Miles + Harbor + mini-swe-agent historical snapshot.
Inspect the launch configuration, model and parser settings, TITO-v2 session
metadata, rollout records, and frozen run metrics.

Do not launch another training run, start SGLang, create sandboxes, download a
model or dataset, use the network, or modify the historical snapshot.

## Deliverable

Write one concise UTF-8 diagnosis to:

```text
$AUTORLBENCH_RESULTS/agent_answer.md
```

State the health verdict, identify the concrete configuration root cause, and
explain the causal chain from that setting through the observed session
mismatches to the training-leaf accounting. Quote the exact logical-rollout,
training-leaf, inflation, fan-out, and mismatch values from the frozen metrics.
Distinguish the configuration defect from any separate residual MiniSWE
behavior; do not merge them into one unproven root cause.
