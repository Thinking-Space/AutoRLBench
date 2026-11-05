# Canonical solution oracle

## Root cause

`Tau2Runner._clean_messages` deletes the user simulator's own tool traffic, and
`_make_completion_fn` selects that path for user requests. Tau2 has already
performed perspective conversion and ownership checks in `UserState.flip_roles`.

## Required remediation

Preserve tool-call/result pairs, order, IDs, and the caller's input. Keep the
OpenAI-compatibility cleanup for null `tool_calls`. The reference removes the
user-specific filtering and updates its call site. Equivalent behavior passes.

## Machine grading

The harness runs `python3 launch.py` after a nonempty `report.md` is submitted.
The final horizon is 16 attempted trajectories with at least 12 error-free
trajectories. Requests must exercise user tools. The cumulative tool-history
retention ratio must be at least 0.99 and at most 1.0, and must match the recorded
request counters. Hidden completion-entry checks cover preservation, null-field
cleanup, empty histories, multiple tool calls, both roles, and input immutability.
The verifier reads only the newly produced experiment logs.

The reference source is the production change in AReaL PR #1759. Reward and
turn-balance measurements accompany the behavioral calibration.
