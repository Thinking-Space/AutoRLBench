# megatron 0.18 passes `name=` kwarg that miles_plugins HuggingfaceAttention does not accept

Training Qwen3.5-9B with arena-rubric GRPO. megatron 0.18's `transformer_layer.py`
calls `build_module(..., name="self_attention", ...)` but `miles_plugins`'
`HuggingfaceAttention.__init__` has no `name` parameter, so `get_model()` raises
`TypeError` on every worker before training starts.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Infrastructure |
| **Domain** | Arena-Rubric RL |
| **Tags** | `megatron` `miles_plugins` `version-mismatch` `TypeError` `build_module` |
| **Framework** | miles (GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | < 1 minute (crash at model construction) |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers. The agent sees only `workspace/`
and the evaluation's `instruction.md`.

Evidence is reconstructed from session documentation (jobs 1560704, 1573326,
1575432 on Cybertron; original cluster logs unavailable). The defect and stack
are real; the freeze is a faithful synthetic reproduction of the error.

## Difficulty

The defect is in the interaction between two installed packages. `megatron`
0.18's `transformer_layer.py:363` calls:

```python
build_module(spec, config=config, layer_number=layer_number, name="self_attention")
```

while `miles_plugins/models/hf_attention.py::HuggingfaceAttention.__init__`
accepts only `(self, config, layer_number=None)` — no `name` parameter and no
`**kwargs`. The result is a `TypeError` raised inside `get_model()` on every
training worker.

Nothing in the launch logs predicts this. The config echoes correctly, the
checkpoint path resolves, and Ray starts cleanly. The crash appears only after
worker processes try to construct the model, and the full traceback is buried in
the Ray worker stderr rather than the driver stdout.

The defect has two plausible wrong diagnoses:
- Blaming the checkpoint path or model config (`--spec` argument) — wrong: the
  crash is a Python TypeError at object construction, not a file-not-found or
  config mismatch.
- Blaming the SGLang version — wrong: the crash happens in Megatron's training
  worker path, before SGLang is ever started.

The fix is a `sitecustomize.py` kwarg-filter wrapper on `build_module` that
silently drops kwargs the target `__init__` does not accept. It must be loaded
via `PYTHONPATH` before every worker process starts, not in the driver — this
is what the per-worker `sitecustomize` mechanism guarantees.

## Reference solution

- **issue**: name `miles_plugins/models/hf_attention.py::HuggingfaceAttention.__init__`
  as the defective symbol (missing `name` parameter), and
  `megatron/core/transformer/transformer_layer.py::build_module` as the caller
  that passes `name=`; connect these to the `TypeError` in the worker traceback.
  Reject checkpoint, SGLang, or Ray config as the cause.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; the fix is a runtime hotpatch with no
measurable training metric effect that could be threshold-graded.
