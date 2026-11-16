# Third-party source

| Location | Origin | License |
| --- | --- | --- |
| `vendor/miles/`, `vendor/miles_plugins/` | Miles `0.2.1.dev1609+g48f39cb4` Python source | Apache-2.0, `licenses/miles.LICENSE` |
| `vendor/megatron/core/`, `vendor/megatron/training/` and siblings | Megatron-Core `0.18.0` Python source | per-file headers, `licenses/megatron-core.LICENSE` |
| `vendor/megatron/bridge/` | Megatron-Bridge `0.5.0` Python source | Apache-2.0, `licenses/megatron-bridge.LICENSE` |
| `vendor/megatron/energon/` | Megatron-Energon Python source | `licenses/megatron-energon.LICENSE` |
| `scripts/convert_hf_to_torch_dist.py` | adapted from Miles `tools/convert_hf_to_torch_dist.py` | Apache-2.0 |
| `polars`, `polars-runtime-32`, `mbridge` | wheels installed by the prepare stage, pinned in `../environment/dependencies.json` | their own licenses |

Per-file copyright headers in the vendored source are authoritative. Code that only served the source
organisation's internal data loader, metrics exporter and an unreleased model family was removed, and three
small compatibility edits were made; `../environment/source_manifest.json` records the counts and the
original and shipped hashes of the edited files. `runtime/fully_async.py` and `runtime/progress_watchdog.py`
are the generic parts of the original runs' rollout producer; everything else under `runtime/` and
`scripts/` other than the conversion script is written for this workspace (Apache-2.0, `LICENSE`).
