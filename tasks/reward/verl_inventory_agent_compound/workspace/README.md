# Inventory GRPO experiment

The workspace contains the veRL training source, launch plan, original logical inputs and
selected native artifacts for a 30-update run with an intermediate checkpoint at step 10.
The effective configurations and phase overrides should be read together.

* `outputs/metrics.jsonl` is the full training/validation metric history.
* `outputs/train.log` contains both native phases. Selected full minibatches and
  validation batches are under `outputs/rollouts` and `outputs/validation`.
* `outputs/tool_events` captures sampled interaction records. Use the order identity,
  requested quantity and request ID when interpreting a tool transaction.
* `outputs/resume` contains checkpoint parameter identities and sampled optimizer
  and parameter statistics from the first resumed step.
* `data_evidence` contains the historical source rows, including parent and
  occurrence identifiers. `environment/prepare.sh` materializes these rows for reruns.

A successful inventory task requires the requested quantity and no other reserved SKU
in the final ledger, plus a valid matching receipt. Report training return and strict
success separately. The intended progress reward measures net inventory progress.
A transport retry refers to the same logical command, and an optimizer restart
preserves the identity of each parameter's saved state.
