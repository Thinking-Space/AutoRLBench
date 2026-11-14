# Environments

Every task runs in one of the **eight fixed shared images** below — digest-pinned, publicly
pullable, and maintained by the benchmark. A task submitter must run their buggy run inside
one of these environments: the recorded evidence shipped in `workspace/` (logs, metrics, W&B)
comes from a run executed in that image, and `task.toml` pins the same image. Only when the
defect reproduces in none of the eight may a task ship its own environment instead — as a
separate environment configuration submitted with a written justification (see below).

In `task.toml`, reference the image by its digest-pinned pullable reference from the table,
plus the row's stack label:

```toml
[environment]
image = "verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6"
image_stack = "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"
```

## The eight shared images

A row is "built" when the reference resolves to a complete, digest-pinned image anyone can
`docker pull` anonymously. The `TCR reference` column is the same image mirrored on a registry
reachable from mainland China.

| `image_stack`                                                               | Pullable reference                                                                                                                                    | TCR reference                                                                                                  |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `miles-0.1.1-sglang0.5.20.dev59-torch2.13.0-flashattn4.0.0b19-cu129-py3.12` | `ghcr.io/merak0514/autorlbench-justrl2@sha256:8d2eb1fe292a4e381915734b7854213002ca9bae6b50efab1bd8cc75cad14a8b`                                       | `ccr.ccs.tencentyun.com/autorlbench/miles-0.1.1-sglang0.5.20.dev59-torch2.13.0-flashattn4.0.0b19-cu129-py3.12` |
| `verl-vllm0.24.0-torch2.11.0-flashattn2.8.3-cu130-py3.12`                   | `verlai/verl:vllm024.dev2@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543`                                                    | `ccr.ccs.tencentyun.com/autorlbench/verl-vllm0.24.0-torch2.11.0-flashattn2.8.3-cu130-py3.12`                   |
| `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12`         | `slimerl/slime:v0.3.2-cu129@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c`                                                  | `ccr.ccs.tencentyun.com/autorlbench/slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12`         |
| `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`                   | `verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`                                                  | `ccr.ccs.tencentyun.com/autorlbench/verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12`                   |
| `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12`                         | `ghcr.io/yhyayi/autorlbench:vllm0.11-torch2.8-cu128@sha256:6f7c302c0161a6c32d978faade165c86c153575baf028e729d7eebe44c06f683`                          | ccr.ccs.tencentyun.com/autorlbench/vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12                           |
| `verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10`                     | `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2@sha256:8550dd839d858b0b8097fe3d3fe5b8ee0b8906469f60c7a6cd55e1d572d070ff`                         | ccr.ccs.tencentyun.com/autorlbench/verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10                       |
| `verl-vllm0.7.4-torch2.5.1-flashattn2.7.4-cu124-py3.10`                     | `hiyouga/verl@sha256:864124d32147f5e9b31916656b441f19d77c1c03e3448c8504b0025df59121b2`                                                                | `ccr.ccs.tencentyun.com/autorlbench/verl-vllm0.7.4-torch2.5.1-flashattn2.7.4-cu124-py3.10`                     |
| `vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10`                          | `ghcr.io/yhyayi/autorlbench:vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10@sha256:f5dbe597c6dad59dcbdee5e363e645f96656f9bea94641f3438f8b2f639c5b3d` | `ccr.ccs.tencentyun.com/autorlbench/vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10`                          |

## Shared image vs. `prepare.sh`

The environment splits along one line: **what cannot be swapped at runtime is baked into the
image; everything else happens in the task's** `prepare.sh`**.**

**The image** is the immutable ground truth shared by every task that names it: compiled
packages, the torch+CUDA ABI, the base image. The eight images above are a fixed set maintained
by the benchmark: tasks never modify them, and a task never adds a ninth. When the recorded
run's stack does not exactly match a row, pick the closest row and bridge the gap in
`prepare.sh` — and re-run the buggy run in that image, so the workspace's recorded evidence
comes from the environment the task declares.

**The exception:** if the defect reproduces in none of the eight — the bug itself lives in a
baked-in layer, say — the task may ship its own environment instead. The submitter
contributes a separate environment configuration: an image they built and pushed to a public
registry, pinned by a single digest-pinned `FROM` line, plus an explanation of why no shared
image fits. The maintainers review these exceptions; an environment the closest row could
carry via `prepare.sh` does not qualify.

### Accepted exception environments

| `image_stack` | Digest-pinned reference | Why no shared image fits |
| --- | --- | --- |
| `areal-v2.0.0-sglang0.5.10.post1-torch2.9.1-flashattn2.8.3-cu129-py3.12` | `ghcr.io/areal-project/areal-runtime:v2.0.0-sglang@sha256:7ce1a21eee7cbdba6d1ac3d5cb5e1162edc392b8d499c9d5d7e17a1cf79276e3` | AReaL v2.0.0 decoupled PPO runs on SGLang 0.5.10.post1 with its sgl-kernel layer compiled against torch 2.9.1+cu129; no shared row carries this ABI (closest: torch 2.8.0 / 2.10.0). The AReaL framework itself is pure Python, but the recorded evidence (frozen 87-step runs, per-token dumps) comes from this image. Used by `areal_kpop_binary_kl_fp32_nan_mask` (PR #35) and `areal_icepop_mask_window_too_narrow` (PR #34). |

`prepare.sh` (the task's `environment/` directory) is the only networked stage. It runs in a
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

So the decision rule when packaging a task: pick the closest row in the table, run the buggy
run in that image, and do the rest in `prepare.sh`. Framework or dependency versions the chosen
image does not carry are layered on per task as above — not baked into a new image. A dedicated
environment is the exception that needs a justification, never a shortcut for a stack the
closest row could carry.
