# Framework source navigation

`train_async_entrypoint.py` is the Miles asynchronous training driver that
`scripts/native_entrypoint.py` executes inside a local Ray cluster. It creates
the rollout manager (SGLang engines and router), the Megatron actor, performs
the initial weight sync, and then overlaps rollout `k+1` with training on
rollout `k`, syncing weights after every update.

| Location | Source |
| --- | --- |
| `train_async_entrypoint.py` | Miles `train_async.py` driver |
| `../fully_async.py`, `../progress_watchdog.py` | continuous rollout producer used by the original runs, generic parts only |
| `../../vendor/miles/`, `../../vendor/miles_plugins/` | Miles `0.2.1.dev1609+g48f39cb4` Python source |
| `../../vendor/megatron/` | Megatron-Core `0.18.0+caeabd765` and Megatron-Bridge `0.5.0` Python source |
| `../../THIRD_PARTY_NOTICES.md`, `../../licenses/` | ownership and license notices |

Compiled dependencies (CUDA, Transformer Engine, FlashAttention, SGLang
kernels) come from the task image; bundled Python source takes precedence on
`PYTHONPATH`.
