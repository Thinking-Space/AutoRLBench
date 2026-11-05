# Canonical issue oracle

Reconciled from the pinned AReaL source, the production diff in PR #1759, the
pinned tau2 role conversion, and the paired telecom rollout recordings.

Expected verdict is **unhealthy**.

## Root cause

- `areal/examples/tau2/agent.py`, `Tau2Runner._clean_messages` and its
  `_make_completion_fn` call site
  - kind is `symbol`
  - observed behavior is that `for_user=not is_agent` enables a branch which
    discards every `role=tool` result and strips nonempty `tool_calls` from
    assistant messages before forwarding them to the user simulator
  - expected behavior is to preserve the simulator's tool-call/result pairs
    while continuing to remove `tool_calls` when its value is null and keeping
    the input messages unchanged

`tau2/src/tau2/user/base.py`, `UserState.flip_roles`, already converts the history
into the simulator's perspective. User messages with tools become assistant tool
calls. Tool results are admitted only for `requestor=user`; agent tool traffic is
rejected. Consequently the removed turns belong to the simulator itself.
Identifying only one erasure or omitting this ownership relationship merits
partial root-cause credit.

## Observable impact

The defective collection attempted 16 trajectories. Fifteen completed without a
runtime error, with mean reward 0.0. Across the completion requests, 42,413 tool
result occurrences and 42,413 tool-call occurrences were present in incoming
user histories; all were absent from the outgoing user histories. These are
cumulative history occurrences, with older turns counted again on later requests.

Repeated user tool calls consume the orchestrator's shared `max_steps=200`
budget. Successful evidence must connect this message loss to simulator
repetition and reduced opportunities for agent actions. Reward alone earns
limited evidence credit. Frozen trajectory and request excerpts permit direct
inspection of the lost observations.

One trajectory also encountered a context-length error. Its error is separately
recorded; the tool-history defect is present in the error-free trajectories.
The staged scope is real dual-model rollout evaluation feeding an RL workflow.

## Trigger conditions

- telecom with user-side tools and `solo_mode=false`
- the user-side completion path through the pinned AReaL runner
- an interaction where the simulator has already called a user tool

The 30-tool telecom user environment supplies the required path. Shipped upstream
airline examples have no user tools, which explains the limited coverage.

## Documentation reconciliation

Changing reward thresholds, model quality, learning rate, or the shared step
budget leaves the history erasure in place. Disabling user tools or selecting
another domain changes the evaluated scenario. The context-length error explains
its own failed trajectory, while the request pairs establish the separate silent
history loss. GRPO optimization metrics are outside this recorded rollout scope.
