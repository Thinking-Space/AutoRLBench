# Training update investigation

An actor/critic PPO replay reached the training stage but did not finish its
first update. Inspect the framework source, launch settings and the frozen
observations under `observed/`. Determine what caused the failure and describe
the smallest repair that preserves the intended computation.

Write `report.md` with a verdict, the responsible code path and a causal chain
connecting the inputs, worker layout, control flow and observed progress.
Explain which workers are affected and why, and distinguish the initiating
failure from later communication failures. Evaluate plausible alternative
explanations against the available evidence. Include the concrete local data
shapes and process-group memberships that support your conclusion.

The configured objective, batch, replay inputs, parallel layout and numerical
semantics are fixed. An arbitrary timeout increase or skipping training does
not satisfy the task. The JSON replay contains authored trajectories; it is
not an online rollout or a policy-quality benchmark.

Do not use the network or launch full GPU training. You may inspect source and
run bounded local calculations. If a repair is requested by the evaluator,
edit the framework and explain it in the same report; the evaluator performs
fresh execution separately. Do not edit the frozen observations.
