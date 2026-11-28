# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective and control workspaces.

Expected verdict: **unhealthy**.

## Root cause

- `checkpoint.py::load_hf_weights`
  - kind: `symbol`
  - observed: after the model is converted to BF16, the loader copies FP32 `source_bias` directly into the existing BF16 `expert_bias` destination; eight source values near 40 collapse to one rounded value.
  - expected: before copying, traverse every model chunk and call `_maintain_float32_expert_bias()` on each module that provides it, preserving an FP32 destination during the Bridge-style copy.

## Observable impact

- Primary source evaluation metric: `eval/correct_action_probability`, 0.02040 at step 12 versus 0.95893 in the control.
- The defective final router is `torch.bfloat16`, has 1 unique value, has a hash different from the FP32 source, and selects expert 0; the control is `torch.float32`, has 8 unique values, matches the source hash, and selects expert 7.
- The defective arm still rises from 0.01799 to 0.02040; “training did not improve” is therefore a wrong attribution.

## Trigger conditions

- A model conversion changes the MoE correction-bias destination buffer to BF16 before FP32 Hugging Face correction biases are copied into it.
- Biases near 40 differ by less than BF16 spacing, so the destination loses their ordering during the copy.

## Documentation reconciliation

- Historical production-scale MoE facts are provenance for the same numerical mechanism, not additional root causes.
- Timezone, router-update-rate, auxiliary-loss, batching, OOM, and other historical changes are unrelated to this reproduction and must not be credited as causes.
