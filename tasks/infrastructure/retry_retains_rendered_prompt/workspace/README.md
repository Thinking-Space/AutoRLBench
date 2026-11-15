# Native Miles training workspace

This workspace provides a public, reduced-size training workload using Miles, Ray and Megatron. Use the report from the current invocation to determine which initialization, generation and training stages actually executed. The default execution is validated; the environment verification manifest records its scope.

## Launch

Run from this directory in the prepared task environment with one visible CUDA GPU and CPU capacity for Hugging Face generation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/train_reproduction.py
```

The default backend is `hf-cpu`: actual Hugging Face model inference runs on the CPU, and the native Miles Ray/Megatron trainer uses one GPU. Miles's `--debug-train-only` option skips SGLang servers and actor-to-generator weight synchronization. The configured training path retains real Ray actors, GRPO loss, forward/backward and optimizer updates. The CPU generation model stays at initialization throughout the probe. Execution scope is recorded in the environment verification manifest.

To inspect the launch arguments without starting execution:

```bash
python3 -B scripts/train_reproduction.py --print-command
```

Optional SGLang generation needs two visible CUDA GPUs, one for training and one for inference:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/train_reproduction.py --generation-backend sglang
```

The SGLang path is unverified: its router did not become ready, and the attempt failed before training. Its configured weight-update interval of 100 would also leave generation weights unchanged between actor updates during this short probe. Use the current invocation report to establish execution scope for either backend. Pure CPU training is unsupported.

The task harness runs `../environment/prepare.sh` to install pinned Python dependencies into its prepared directory and create model assets under `AUTORLBENCH_WORKSPACE/assets/public-qwen2`. Run through the prepared environment so its dependency path is available. Assets can also be regenerated with `scripts/prepare_public_assets.py --dest assets/public-qwen2` after dependencies are installed.

The supported platform image alias is `autorlbench-slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12`. Bundled Python framework source is used together with the image's compiled CUDA, Transformer Engine, FlashAttention and SGLang dependencies. Compatibility and execution scope must be checked against the current run report.

## Navigation

| Location | Contents |
| --- | --- |
| `scripts/` | Launch, local Ray bootstrap and asset preparation |
| `runtime/` | Workload configuration and public adapters |
| `runtime/source/` | Archived framework entrypoint and source provenance |
| `vendor/` | Bundled Miles, Miles plugins and Megatron source |
| `assets/public-qwen2/` | Reproducible model, tokenizer and asset metadata |
| `experiment_logs/<invocation>/` | Launch record and evidence produced by a new invocation |
| `logs/historical/` | Separately recorded, redacted historical evidence |
| `THIRD_PARTY_NOTICES.md` and `licenses/` | Source and license notices |

The reduced workload uses a randomly initialized public Qwen2 architecture with 361,600 parameters and seed 20261115, public color-token messages, and a score computed from generated token IDs. Private checkpoints, datasets, prompt payloads and reward services are excluded. Training is delegated to the native Miles Ray/Megatron path with GRPO; there is no replacement training loop. The reduced workload is not a replay of the historical model, corpus or training curve.

Historical records are indexed by `logs/historical/manifest.json`. The archives `master-0.jsonl.gz` and `worker-0.jsonl.gz` preserve 76,899 and 275 API record positions respectively, with private payloads replaced by redaction markers. `metrics/{step,rollout,perf}/*.json` contains 160 step, 20 rollout and 20 performance records with numeric values preserved. This is the full log API result for the selected training pods, not every Ray worker, inference server or external service file. Historical metrics do not describe a new invocation.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `../environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The default native failure/reference validation uses the resulting source.
