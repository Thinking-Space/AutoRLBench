# Canonical issue oracle

Reconciled from the sanitized rollout logs, the historical source lifecycle,
and the focused retry regression in the workspace.

Expected verdict: **unhealthy**.

## Root cause

- `runtime/generate.py::run_generation`
  - kind: `symbol`
  - observed: accepts a messages list and renders it, but then replaces
    `sample.prompt` with the rendered string.
- `runtime/sample.py::Sample.reset_for_retry`
  - kind: `symbol`
  - observed: clears generated state while deliberately preserving
    `sample.prompt`.
- `runtime/fully_async.py::clone_group_for_retry` and
  `reset_source_group_for_recycle`
  - kind: `symbol`
  - observed: use `reset_for_retry()` for same-prompt retry and environment
    recycling without restoring the original messages.
- `runtime/generate.py::validate_messages`
  - kind: `symbol`
  - observed: rejects the preserved string on replay with
    `native message input required`.

These symbols form one retry-state root cause: generation mutates the durable
prompt field from its input representation to its rendered representation,
while reset preserves that field. The model, network, GPU, environment
session, and checkpoint loader are not causes of the repeated local input
failure.

## Observable impact

- The recorded run completed 20 rollouts and 160 optimizer updates.
- Rollout raw reward stayed in the finite range 0.53516 to 0.63281 and the
  last normal update reached step 159.
- The retry path then logged `native message input required` repeatedly.
- The same loop reported 128 consecutive aborted or recycled groups and
  terminated with `refusing to wait forever`.
- A focused repair rebuilt the original messages before generation. It passed
  lifecycle tests for same-prompt retry, environment recycle, and direct input
  of rendered text; 56 tests passed.

## Trigger conditions

- Generation requires native messages at entry.
- The sample stores the rendered prompt in the shared chat-template field and
  the same object is reused for retry or recycling.
- The retry reset preserves that field.
- A replay occurs before a fresh sample builder can replace the field.

## Documentation reconciliation

The later `aborted group` counter and final `refusing to wait forever` error
are downstream symptoms. The exception emitted during each replay is the
direct evidence. Because the failure occurs before a service request, a
diagnosis that depends on an unavailable model server, proxy, DNS, or GPU is
not supported.
