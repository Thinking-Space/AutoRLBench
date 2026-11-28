# Canonical solution oracle

## Root cause

- `checkpoint.py::load_hf_weights`
  - kind: `symbol`
  - observed: FP32 correction bias is copied into a BF16 destination after model conversion, collapsing the router ordering.
  - expected: restore expert-bias buffers to FP32 before the copy.

## Observable impact

- `eval/correct_action_probability` ends at 0.02040 rather than 0.95893.
- The defective final router is BF16 with 1 unique value, a source hash mismatch, and expert 0; the control is FP32 with 8 values, a source hash match, and expert 7.
- The packaged logs, metrics, and summaries for both arms come from the official shared-image A/B in Slurm job 23051184, using the task's pinned image digest and imported SQSH SHA-256 `9a3c9ab63f30a5b524744dc56173fe3e37d99e9a1c2dbafde307733e3ef67819`.
- `provenance/` preserves the exact submission script, image-import output, and Slurm accounting row. `control_patch/checkpoint.py` is the complete repaired file; `historical_evidence_manifest.json` hashes both source sides, every recorded metric artifact, and all provenance files.

## Trigger conditions

- The model wrapper converts router buffers to BF16 before a Bridge-style FP32 source copy, at a value scale where BF16 spacing exceeds the source spread.

## Required remediation

- `checkpoint.py`: immediately before copying, traverse every model chunk and call `_maintain_float32_expert_bias()` on modules that provide it. Do not include unrelated historical fixes.

Machine grading reads only the fresh run. At step 12 it requires probability at least 0.955 and at most 1.0, FP32 router bias, 8 unique values, source/router hash equality, and selected expert 7; `train.py` must retain its packaged SHA-256.
