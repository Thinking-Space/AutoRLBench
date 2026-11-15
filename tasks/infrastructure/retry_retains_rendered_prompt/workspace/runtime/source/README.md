# Framework source navigation

This directory contains archived source used by the native Miles launch. Start with `../../README.md` for environment requirements, commands and output locations. Inspect the current invocation's report for execution scope; execution scope is recorded in the environment verification manifest.

`train_async_entrypoint.py` is the archived async training driver executed by the local Ray bootstrap. Additional workload modules are in the parent directory (`../`).

The default launch uses one GPU for the native Miles Ray/Megatron trainer and actual Hugging Face CPU generation (`--generation-backend hf-cpu`). Miles's `--debug-train-only` option skips SGLang servers and actor-to-generator weight synchronization; the configured path retains native Ray actors, GRPO loss, forward/backward and optimizer updates. The CPU generation model remains at initialization. The default native failure/reference pair is validated; execution scope is recorded in the environment verification manifest.

Optional `--generation-backend sglang` requires two GPUs. That path remains unverified: the router did not become ready, and the attempt failed before training. Its configured weight-update interval of 100 would likewise keep generation weights unchanged between actor updates during the short probe. Inspect the current report for actual execution scope.

## Source inventory

| Location | Recorded source |
| --- | --- |
| `train_async_entrypoint.py` and archived parent-directory modules | Historical overlay revision `5717feb722dbde9fb5334430ed04a27d861b79cf` |
| `../../vendor/miles/` and `../../vendor/miles_plugins/` | Miles `0.2.1.dev1609+g48f39cb4` wheel Python source |
| `../../vendor/megatron/` | MCore `0.18.0+caeabd765`, Bridge `0.5.0` and auxiliary source recovered from an inspected image |
| `../../../environment/runtime_sources.json` | Per-file source pins, original/shipped hashes and sanitization records |
| `../../THIRD_PARTY_NOTICES.md` and `../../licenses/` | Ownership, licensing and retained notice provenance |

The historical wheel and the inspected image are separate source records. Bundled Python source does not establish that compiled dependencies match the historical environment, and inclusion of a module does not establish that it executed. CUDA, Transformer Engine, FlashAttention and SGLang binaries remain environment dependencies.

The public workload replaces private model assets, data, generation integration and reward services with reproducible public components. It adds scheduling configuration and observational hooks while delegating training to the native framework. Source examples and paths were sanitized; private-to-public mappings are not distributed. Per-file copyright and license headers remain authoritative.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `../../../environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The default native failure/reference validation uses the resulting source.
