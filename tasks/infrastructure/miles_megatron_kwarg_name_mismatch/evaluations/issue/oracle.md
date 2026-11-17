# Canonical issue oracle

Reconciled from the task's frozen evidence and session-documented error traces
(jobs 1560704, 1573326, 1575432; evidence is a faithful synthetic reconstruction
from the documented TypeError; original cluster logs unavailable).

Expected verdict: **unhealthy**.

## Root cause

- `miles_plugins/models/hf_attention.py::HuggingfaceAttention.__init__`
  - kind: `symbol`
  - observed: the method signature is `__init__(self, config, layer_number=None)` — it
    accepts neither a `name` keyword argument nor `**kwargs`. megatron 0.18's
    `megatron/core/transformer/transformer_layer.py:363` calls:

    ```python
    build_module(spec, config=config, layer_number=layer_number, name="self_attention")
    ```

    This raises `TypeError: HuggingfaceAttention.__init__() got an unexpected keyword
    argument 'name'` on every training worker inside `get_model()`, before any
    checkpoint is loaded or any rollout is generated.
  - expected: `HuggingfaceAttention.__init__` accepts (or silently ignores via `**kwargs`)
    the `name` keyword argument that megatron 0.18 passes, so `get_model()` succeeds and
    training starts normally.

## Observable impact

- Primary source metric: job exit code. All training workers exit with code 1
  immediately after `get_model()` is called, before any training step or rollout.
- Worker stderr contains a traceback ending in:
  `TypeError: HuggingfaceAttention.__init__() got an unexpected keyword argument 'name'`
- Driver stdout shows no completed rollout step and no reward curve.
- The launch config echoes `--spec miles_plugins.models.qwen3_5` correctly, so the
  error cannot be diagnosed from the config log alone.

## Trigger conditions

- megatron version 0.18 (ships `name=` kwarg in `transformer_layer.py:363`)
- miles_plugins `HuggingfaceAttention` without a `name` parameter or `**kwargs`
- Both packages installed in the same Python environment (the shared miles image
  ships both via `dist-packages`)

## Documentation reconciliation

- "The checkpoint path is wrong" is a wrong attribution: the crash happens at
  model construction, before any file read or checkpoint load.
- "The SGLang version is incompatible" is a wrong attribution: the crash is in
  Megatron's training worker path; SGLang is not started until after `get_model()`
  succeeds.
- "The `--spec miles_plugins.models.qwen3_5` argument is incorrect" is a wrong
  attribution: the spec is correctly resolved; the crash is a Python `TypeError`
  at instantiation time, not a missing module.
- The `name` argument in megatron 0.18 is used only for module naming and
  diagnostics; silently dropping it via a kwarg filter does not affect training
  correctness.
