# Environments

Every task runs in one packaged, publicly pullable image. **Before adding a new environment,
check the table below** — an image that already matches your recorded stack is the fastest path
to a valid task, and reuse keeps the benchmark's ground-truth environments from multiplying.

In `task.toml`, reference the image by its digest-pinned pullable reference, plus an optional
stack label describing what is inside it:

```toml
[environment]
image = "verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6"
image_stack = "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"
```

## Built images

A row is "built" when the reference resolves to a complete, digest-pinned image anyone can
`docker pull` anonymously. `Tasks` counts the tasks currently using it (0 = reusable spare).

| `image_stack` | Pullable reference | Stack | Tasks |
| --- | --- | --- | --- |
| `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` | `verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6` | vLLM 0.17.0 · torch 2.10.0 · cu129 (veRL travels with the workspace) | 10 |
| `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` | `slimerl/slime:v0.3.2-cu129@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c` | slime 0.3.2 baked in · sglang 0.5.15 · cu129 | 3 |
| `verl-vllm0.7.4-torch2.5.1-flashattn2.7.4-cu124-py3.10` | `hiyouga/verl@sha256:864124d32147f5e9b31916656b441f19d77c1c03e3448c8504b0025df59121b2` | vLLM 0.7.4 · torch 2.5.1 · cu124 (veRL 0.3.0.post1 / Ray 2.40.0 travel with the workspace) | 1 |
| `vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10` | `ghcr.io/yhyayi/autorlbench:vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10@sha256:f5dbe597c6dad59dcbdee5e363e645f96656f9bea94641f3438f8b2f639c5b3d` | vLLM 0.6.3 · torch 2.4.0 · cu121 (Search-R1 lineage) | 1 |
| `verl-vllm0.24.0-torch2.11.0-flashattn2.8.3-cu130-py3.12` | `verlai/verl:vllm024.dev2@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543` | vLLM 0.24.0 · torch 2.11.0 · cu130 (veRL travels with the workspace) | 1 |
| `justrl2-data-relocation` | `ghcr.io/merak0514/autorlbench-justrl2@sha256:8d2eb1fe292a4e381915734b7854213002ca9bae6b50efab1bd8cc75cad14a8b` | JustRL-II / miles · ray · cp2dp2 | 1 |
| `verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10` | `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2@sha256:8550dd839d858b0b8097fe3d3fe5b8ee0b8906469f60c7a6cd55e1d572d070ff` | vLLM 0.8.5.post1 · torch 2.6.0 · cu124 (veRL travels with the workspace) | 1 |

Six more environments exist as build recipes but are not pushed yet (`verl-v061-grpo`,
`verl-vllm011`, `trl-grpo-torch26`, `verl-0.9-vllm0.28`, `tool-rl-vllm024`, `maestro-vllm011`);
see the "Exceptions" section of [docker/README.md](docker/README.md) for their status and how to
finish them.

## Dockerfile vs. `prepare.sh`

The environment splits along one line: **what cannot be swapped at runtime is baked into the
image; everything else happens in the task's `prepare.sh`.**

**The image** (`docker/images/<name>.Dockerfile`, a single digest-pinned `FROM`) is the immutable
ground truth shared by every task that names it: compiled packages, the torch+CUDA ABI, the base
image. Never modify an existing Dockerfile for one new task — that silently rewrites the
environment of the tasks already using it. A new stack means a new one-line file, named after
what is actually baked in (slot grammar in [docker/README.md](docker/README.md#2-one-image-per-environment-shared-by-tasks)).

**`prepare.sh`** (the task's `environment/` directory) is the only networked stage. It runs in a
throwaway container of that image, with `$AUTORLBENCH_WORKSPACE`, `$AUTORLBENCH_MODELS`,
`$AUTORLBENCH_DATASETS` and `$AUTORLBENCH_PREPARED` writable, before the agent or verifier ever
start. Its jobs, in rough order of frequency:

1. **Fetch assets** — models and datasets via the task-local fetcher (`fetch_hf.py`), pinning
   repo, revision, and sha256. Example:
   [verl_taco_difficulty_mismatch](tasks/data/verl_taco_difficulty_mismatch/environment/prepare.sh).
2. **Verify staged evidence** — some tasks diagnose a frozen snapshot and only check the files
   are present and non-empty. Minimal example:
   [trl_gsm8k_eos_mismatch](tasks/configuration/trl_gsm8k_eos_mismatch/environment/prepare.sh).
3. **Install or patch pure-Python layers** — `pip install --target` (or a patched copy of an
   installed package) into `$AUTORLBENCH_PREPARED/pysite`, exposed to later stages through a
   `PYTHONPATH` line in `$AUTORLBENCH_PREPARED/env`. This is how a task gets framework or
   dependency versions the shared image does not carry, without touching the image. Example:
   [verl_moe_token_normalization](tasks/infrastructure/verl_moe_token_normalization/environment/prepare.sh),
   which copies the image's megatron out, patches it, and stages it under `$AUTORLBENCH_PREPARED`.

What a task needs in between — a couple of run-time variables — goes in `[environment.env]` in
`task.toml`, not into any of the above.

So the decision rule when packaging a task: if the recorded run's stack matches a row in the
table, reuse it and do the rest in `prepare.sh`. Only when no baked-in layer (torch build, CUDA,
compiled kernel, engine version) matches do you add a new environment — and then build it, push
it somewhere publicly pullable, and pin the digest in `task.toml`.
