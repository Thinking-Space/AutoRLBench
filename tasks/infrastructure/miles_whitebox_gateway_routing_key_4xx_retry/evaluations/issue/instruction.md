# Rollout-gateway failure diagnosis

The workspace holds a GRPO training launch and the logs of the run it produced:
the full miles source tree, the rollout-side components the launch drives
(`rollout/`, `recipes/`, `scripts/`), the gateway the rollout talks through
(`gateway/`), the routing proxy in front of the engine (`routing_proxy/`), and
under `recorded_run/` the driver, gateway, and routing-proxy logs of a run that
was stopped after thirty minutes.

Decide what kept this run from making training progress, and find the concrete
root cause or causes in the code. Explain the causal chain from each of them
to what the logs show, with specific file, line, and log evidence. Separate
root causes from symptoms and from unrelated noise.

Things worth knowing before you conclude:

- The run was stopped on purpose after thirty minutes; no step completed
  before that. Whatever you claim must be visible in the logs you have.
- The training stack addresses its engines and routing components by HTTP.
  The same retry helper is used on both sides of the gateway.
- You may run the small components locally to check a hypothesis; they are
  CPU-only and need no GPUs. Do not launch the full training run — it is
  multi-GPU, runs for most of an hour, and adds nothing the logs below do
  not already show. Do not use the network, and do not modify the recorded
  logs.

Write one concise UTF-8 diagnosis to
`$AUTORLBENCH_RESULTS/agent_answer.md`. Do not propose a fix.
