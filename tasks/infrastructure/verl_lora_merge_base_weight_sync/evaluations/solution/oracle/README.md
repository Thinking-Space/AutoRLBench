# Canonical solution oracle

## Root cause

- `verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.get_per_tensor_param`, merged-LoRA branch
  - kind: `symbol`
  - observed: `state_dict()` is captured inside `merged_lora_context`, but the
    DTensors (which alias the live parameters under FSDP2) are only gathered
    with `full_tensor()` by the lazy generator after the context has restored
    the base weights, so vLLM receives base weights at every sync.
  - expected: materialize every merged tensor before the context exits.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`.
- The rollout policy never receives the trained LoRA delta; validation stays at
  the base level (0.0842 / 0.0811 at step 60 in the two defective runs) while
  the actor keeps updating (`actor/kl_loss` 7.8 / 7.3, `rollout_corr/kl` 81.5 /
  74.8 at step 60).

## Trigger conditions

- `model.lora_rank > 0`, `model.lora.merge=True`, `actor.strategy=fsdp2`; every
  sync after the adapter has moved off its zero initialisation.

## Required remediation

- `verl/workers/engine/fsdp/transformer_impl.py`: keep `merged_lora_context`
  open until the consumer has materialized every tensor. `reference.patch` is
  the upstream fix (verl PR #7014, `d45a68d5` -> `6e3c39d6`): the merge branch
  returns a generator, `_merged_lora_per_tensor_param`, that yields
  `full_tensor()` copies (and clones of plain tensors) from inside the context.
  Materializing all tensors into a list inside the context is equivalent at
  this model size. Turning off `lora.merge`, LoRA, or FSDP2 avoids the code
  path rather than repairing it; the rerun pins all three.

## Machine grading and calibration

Machine grading reads only new `experiment_logs` files and requires
`val-core/openai/gsm8k/acc/mean@1 >= 0.4` at step 60 (`score_mode=final`),
with an accuracy cap of 1.0. The step-60 validation dump is re-graded with the
verifier's own copy of verl's GSM8K scorer against a protected answer key: all
1319 test problems must appear exactly once, answers come from the key, and the
re-grade must agree with the logged value within 0.002.

Two defective and two repaired runs calibrate this initial threshold (seed 1
on 1 GPU, seed 2 on 2 GPUs). Step 60: defective 0.08415 / 0.08112, repaired
0.7771 / 0.7794. Maximum over every defective validation point: 0.09098.
Minimum over repaired points from step 5: 0.7263. Repaired step 0: 0.08112 /
0.07961, so a fix without training does not pass. Margins at step 60 are 0.316
above the defective arm and 0.377 below the repaired arm. These are observed
margins over two seeds per side, not variance bounds or a claim of
cross-environment stability.

`control/` holds the repaired seed-1 run's sanitized console history and
W&B history; `control-seed2/` the seed-2 run's console history.
`historical_evidence_manifest.json` lists the defective seed-1 validation
series that the workspace freezes.
