# RL training run diagnosis

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Inspect the source, effective configuration, launch script,
metrics, rollouts, debug probes and frozen evidence. Connect any unusual
implementation to observed behavior rather than treating it as causal by
appearance alone.

Do not launch training, use the network, or modify the historical snapshot.
Write one concise diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the
verdict, concrete root cause and causal evidence. Distinguish root causes from
symptoms; do not implement a fix.
