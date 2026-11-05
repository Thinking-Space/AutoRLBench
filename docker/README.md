# docker/

The environment images every task runs in, and the conventions they follow.

| Path | What it is |
| --- | --- |
| `images/<name>.Dockerfile` | One environment per file. A task selects it with `environment.image = "<name>"` in its `task.toml`. |
| `verl-v061-runtime/` | The pip lock the legacy `verl-v061-grpo` recipe (see [Exceptions](#exceptions)) installs from. |

Everything else that used to live here — the baked activate hook, the fetch tooling, the
per-task docker-compose files — is gone: activation is injected at runtime by the harness
([`autorlbench/runtimes/base.py`](../autorlbench/runtimes/base.py)), fetching travels with each
task's `environment/` directory, and GPU wiring is synthesized from the task's `gpus`
declaration.

## 1. The problem this design solves

This is a benchmark whose grading standard is *reproducing a specific training failure*. That
one sentence drives every trade-off here:

- If the environment drifts, the ground truth being reproduced is silently rewritten — and
  nothing errors.
- So **reproducibility beats uniformity**, and beats "fewer images".
- And a **loud failure beats a silent downgrade**.

The task set also spans several mutually incompatible training stacks (veRL, slime, areal,
miles, …), so one shared image was never on the table.

## 2. One image per environment, shared by tasks

`docker/images/<name>.Dockerfile` defines one environment and may be shared by several tasks.
A task names the environment in `task.toml`:

```toml
[environment]
image = "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"
```

The environment name is `docker/images/<name>.Dockerfile` minus the suffix.

Because an image is shared, its content is the ground-truth environment of **every** task that
uses it. Editing a Dockerfile changes all of them at once; a task that needs different baked-in
dependencies opens a new environment name instead. A task that only misses one or two pure
Python packages installs them itself in `environment/prepare.sh` (see
[section 5](#5-fetch-tooling-travels-with-the-task)) — the change is not visible to the
agent or verifier containers, which is exactly the point.

### Naming convention

The name describes the stack, not the tasks, and is never derived from a content hash. It
encodes **only the layer that is baked into the image and cannot be swapped in `prepare.sh`**:
compiled packages, torch+CUDA ABI, the base image. Pure Python packages that a task can
override with `pip install --target` plus a `PYTHONPATH` line in `prepared/env`
(`transformers`, `tokenizers`, `datasets`, `accelerate`, `ray`, …) do **not** belong in the
name — their exact versions are pinned by the `FROM` digest anyway.

The name is assembled from **fixed slots, in a fixed order**; an absent slot is not
"omitted", it means **that component genuinely is not in the image**:

```
[<framework>-<version>-]<engine><version>-torch<version>-flashattn<version>-cu<version>-py<version>
```

| Slot | Written as | Absent means |
| --- | --- | --- |
| framework (leading) | `slime-v0.3.2` (baked in), `verl-` (family label, see below) | no framework baked in — it arrives bind-mounted with the task's workspace |
| inference engine | `vllm0.11.0`, `sglang0.5.15` | no rollout / inference engine in the image |
| torch | `torch2.11.0` | no torch |
| flash-attn | `flashattn2.8.3` | flash-attn really is **not installed** |
| CUDA | `cu129` | no CUDA (a CPU image) |
| python | `py3.12` | — (essentially always present; this slot carries information, not absence) |

The leading framework slot gets a version **only when the framework is actually baked into
the image** (`slime-v0.3.2`, `slime-v0.5.0rc0`). The veRL family does not bake the framework —
`import verl` fails in the base image; the framework travels with the workspace — so `verl-`
is a versionless family label that groups the veRL tasks, followed by the real engine /
torch / flashattn slots. Examples:

- `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` — no framework slot: veRL comes with
  the workspace.
- `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` — same, carrying the `verl-`
  family label to distinguish it from the line above.
- `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` — framework baked in,
  so it leads with its version; torch / flashattn listed because they really are installed.

Format rules: all lowercase, no `+`. The engine / `torch` / `flashattn` slots use the full
`major.minor.patch`; pure post-release suffixes (`.post1`, `.dev2`) are left to the digest,
but pre-release markers that genuinely fork the stack are kept (`slime-v0.5.0rc0` exists
precisely because `0.5.0rc0 ≠ 0.5.15`). CUDA stays `cuNNN`, python stays `major.minor` —
neither slot promises patch-level fidelity.

Two consequences worth internalizing:

- **"Absent means absent" is a promise.** The `flashattn` / `py` slots describe what is
  actually in the image, *including transitive dependencies* — upstream veRL / slime images
  often carry flash-attn transitively, and leaving the slot out would turn "absent = not
  installed" into a lie. A post-build consistency check (`import flash_attn`,
  `python --version`, `pip show vllm`, asserted against the name) also catches
  "changed the base image, forgot the name".
- **The name makes look-alikes visible, not mergeable.** Two environments that read the same
  are candidates for merging, but whether a merge is real is decided by re-running the tasks
  on the merged image and confirming the scores do not move — never by name similarity alone
  (reproducibility beats uniformity, section 1).

### Why slime has two environments

Not a patch-level relationship — two genuinely different stacks. The retool / search-r1
recipes import `sglang.srt.constants.GPU_MEMORY_TYPE_CUDA_GRAPH`, which does not exist in
`0.5.0rc0`; meanwhile the rlve tasks' recorded results were taken on `0.5.0rc0`, and moving
them forward means rewriting their ground truth under a different torch. Two stacks that
cannot be reconciled get two names.

### Exceptions

A few Dockerfiles predate or deviate from the single-`FROM` rule:

- `miles-ngc2506-distributable.Dockerfile` is a multi-layer **build recipe** (fork tree,
  wheels) for a self-built image — its `FROM` line names a *base*, so pulling it would fetch
  the wrong image. No task uses it yet; it is kept as the record of how that image was built.
- `verl-v061-grpo.Dockerfile` is the veRL 0.6.1 GRPO **build recipe**: a venv layer over the
  digest-pinned upstream vLLM image, with `verl-v061-runtime/` as its lock file. Adapted to
  the bake-nothing convention on 2026-11-06 (no baked tooling or ENV; internal venv paths
  renamed off `/opt/autorlbench-*`). It must be built and pushed as
  `autorlbench/verl-v061-grpo:torch2.8-vllm0.11-cu128` once, then collapses to a single
  digest-pinned `FROM`. Until the push exists, the tasks' image pull fails — build first
  (`algorithm/verl_policy_loss_padded_mask_only`, `algorithm/verl_policy_loss_padded_unmasked`,
  `infrastructure/verl_rollout_weight_overwrite`).
- `tool-rl-vllm024.Dockerfile` is the tool-RL stack captured from the paired Modal runs
  (Python 3.12, torch 2.11.0, vLLM 0.24.0): a **build recipe** to be built and pushed as
  `autorlbench/tool-rl-vllm024:py312-cu130` once, then collapsed to a single digest-pinned
  `FROM`. Adapted to the bake-nothing convention on 2026-11-06. Until the push exists, the
  task's image pull fails — build first (`algorithm/verl_reinforcepp_observation_credit`).
- `trl-grpo-torch26.Dockerfile` is the restored build recipe for the TRL GRPO task pair
  (`algorithm/trl_dapo_accumulation_normalization`,
  `infrastructure/trl_conditional_reward_logging`): no published image carries that stack
  (Python 3.12, torch 2.6.0, transformers 5.15.1), so it must be built and pushed as
  `autorlbench/trl-grpo-torch26:py312-cu124` once, after which it collapses to a single
  digest-pinned `FROM`. Until the push exists, the tasks' image pull fails — build first.
- `verl-vllm011.Dockerfile` is the restored build recipe for the PiSSA GRPO task pair
  (`algorithm/verl_pissa_residual_base_reference`,
  `infrastructure/verl_pissa_vllm_adapter_request`): a thin pip layer over the digest-pinned
  upstream `verlai/verl:vllm011.dev5` image, to be built and pushed as
  `autorlbench/verl-vllm011:vllm0.11-torch2.8` once, after which it collapses to a single
  digest-pinned `FROM`. Until the push exists, the tasks' image pull fails — build first.
- `verl-0.9-vllm0.28.Dockerfile` is the restored build recipe for the verl 0.9.0 GRPO pair
  (`configuration/verl_grpo_ppo_mini_batch_too_small`,
  `reward/verl_grpo_gsm8k_any_number_reward`): a pip + source-built flash-attn layer over
  the digest-pinned `vllm/vllm-openai:v0.28.0` image (torch 2.13.0 / cu130), to be built
  and pushed as `autorlbench/verl-0.9-vllm0.28:torch2.13-cu130` once, after which it
  collapses to a single digest-pinned `FROM`. Until the push exists, the tasks' image pull
  fails — build first. Recorded from the conda env the runs used, never built as an image
  before the recipe was deleted by the 87c6be6c port.

Two further environment gaps, kept pending their task authors (these tasks do not pass
`load_task_package` until the Dockerfiles exist):

- `vl-grpo-cu129-recorded` (task `infrastructure/verl_qwen3_vl_grpo_training_case_v8k2`)
  never had a Dockerfile — the recorded environment was the author workstation's conda
  env, whose full pip freeze is preserved in the task's
  `environment/recorded-runtime-lock.txt` (torch 2.8.0+cu129, vLLM 0.11.0+cu129,
  Python 3.12).
- `idea-issue` (task `reward/idea_awm_reward_diagnosis`) never had a Dockerfile — the task
  README itself marks the definition as pending. The task is CPU-only static diagnosis
  (prepare, verifier tests and reproduce.py are stdlib-only; the judge runs outside the
  sandbox), so the image content is not part of the graded ground truth.

One task keeps a non-image exception file:

- `infrastructure/verl_checkpoint_host_memory_oom` keeps its
  `environment/docker-compose.yaml` — deliberately. It records the task's 20 GiB
  host-RAM ceiling, which `launch.py` asserts against `/sys/fs/cgroup/memory.max` and
  which is part of the defect contract. The current harness does not read task-local
  compose files and has no memory-limit mechanism, so the file is inert until such
  support exists; without the limit the task's own assert fails loudly.

A few recorded environments (`justrl2-data-relocation`, `verl_checkpoint_host_memory_oom`,
`verl-qwen3vl-cu130`) carry descriptive names instead of the slot grammar; treat them as
opaque identifiers.

## 3. Every Dockerfile is one digest-pinned `FROM`

Whatever the name says, each runtime Dockerfile collapses to a single line that pins the
image by digest:

```dockerfile
FROM verlai/verl:vllm017.latest@sha256:4c43bbf1...
```

pointing either at an upstream framework image or at an image we built once and pushed
(when the build is elaborate, it is pre-built and pushed, and the recipe is maintained
separately — as with `miles-ngc2506-distributable`). The test suite enforces the digest pin:
upstream tags move (`vllm017.latest` especially), and a task's recorded results must not
move with them. Changing a digest is a deliberate upgrade — re-run every task on that
environment afterwards.

The image reference is resolved from the `FROM` line by
[`EnvironmentImage`](../autorlbench/core/types.py) (an optional `# autorlbench-image:` header
still wins, for backwards compatibility), and the same reference reaches the Docker backend,
`build-images`, and the Harbor adapter — one source, no drift.

### Pulling

Nothing is built at evaluation time. The Docker backend checks whether the digest-pinned
reference is present locally: present means reuse, missing means `docker pull`. `evaluate`
and `harbor-evaluate` therefore pull on demand, which on a cold machine is a multi-GB hit; to
pay it up front:

```sh
python3 -m autorlbench build-images                 # every environment under docker/images/
python3 -m autorlbench build-images \
    verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 \
    slime-v0.5.0rc0-sglang0.5.0rc0-torch2.8.0-flashattn2.7.4-cu126-py3.12   # a subset
python3 -m autorlbench build-images --rebuild \
    verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12                  # re-pull even if present
```

`--rebuild` (alias `--repull`) fetches the very same digest again — useful only when a local
copy is corrupted. Since nothing is built from a context, there is no staleness to detect:
the digest is the truth.

## 4. Runtime: nothing AutoRLBench-specific is baked in

The image is the upstream stack, full stop. The responsibilities the old baked-in "tooling
block" used to carry moved to the backend and the task:

- **Mount points** (`/autorlbench/{prepared,models,datasets}`): no `mkdir` in the image —
  the Docker backend creates them as bind-mount targets, the local backend uses host paths,
  and Harbor mounts the prepared directory read-only.
- **Working directory**: passed explicitly (`docker create -w` / Harbor's `workdir`), never
  the image's `WORKDIR`.
- **Image-level env** (`CUDA_DEVICE_ORDER`, `TOKENIZERS_PARALLELISM`, …): no image defaults.
  A task that needs them writes them into `prepared/env` in its `prepare.sh`, so they travel
  with the task, not the image.

### Activation is injected at runtime

The activate logic used to be `/etc/profile.d/autorlbench-activate.sh` inside the image,
sourced by a login shell. Its single source is now
`autorlbench/runtimes/base.py:AUTORLBENCH_ACTIVATE_BODY`, and each backend delivers it:

- **docker / local**: the body is prepended to every `sh -c` command the harness runs;
- **Harbor**: prepare writes it to `prepared/autorlbench-activate.sh`, and the agent /
  verifier containers reach it through `BASH_ENV` (Harbor runs a non-login `bash -c`).

Because commands are non-login shells, the image's `ENV PATH` survives untouched; the body
still restores it from `/etc/autorlbench-image-path` *if that file exists* (new images never
create it, so this is inert defensive code). What it actually does:

1. **Wires in prepare's outputs** — reads `$AUTORLBENCH_PREPARED/env` line by line and
   exports each `KEY=VALUE`. The file is parsed as data, not sourced: a task package must
   not gain arbitrary shell execution in every command. It lives under `prepared/`, not the
   workspace, because the agent can write the workspace and the verifier reads this file —
   grading must not consult something the examinee can edit.
2. **Gives ad-hoc installs somewhere to land (agent / rerun stages only)** — sets
   `PYTHONUSERBASE=/workspace/.autorlbench-user`, `PIP_USER=1`, and
   `PIP_BREAK_SYSTEM_PACKAGES=1` (Ubuntu 24.04's system python is externally managed and
   refuses even `--user` installs without it). An agent that `pip install`s writes into its
   own workspace, never the image's site-packages, and the verifier never sees it.

### Stages, mounts, and network

| Stage | What it does | Network | Download caches |
| --- | --- | --- | --- |
| `prepare` | runs the task's `environment/prepare.sh`: fetches weights and datasets, writes `prepared/env` | bridge (the only stage that downloads) | writable |
| `agent` | the agent works in the workspace | per backend policy | not mounted |
| `rerun` | the harness re-runs the experiment per `run_spec.json` | bridge, offline-enforced | not mounted |
| `verifier` | grades this pass's fresh logs | configurable (`--verifier-network`) | read-only |

Which stage sees what is decided by predicates in
[`autorlbench/runtimes/base.py`](../autorlbench/runtimes/base.py), each with a rationale in
its docstring:

- `sees_environment_dir()` — everyone but the agent: `prepare.sh` names the repos and hashes
  a task was assembled from, and any of those names can point at the defect being diagnosed.
- `sees_download_cache()` — prepare and verifier only: an HF cache's directory names are its
  repo names (`datasets--<owner>--<repo>`), and a repo name is sometimes named after the
  defect it carries. The agent and rerun need nothing from the cache: prepare hard-linked
  the real bytes into the workspace.

The offline switches are also per stage (`SandboxRuntime._network_policy_env`): prepare gets
`WANDB_MODE=offline` and nothing else; agent, rerun and verifier get
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE` / the W&B offline set. Any
data those stages use must be what prepare delivered — a silent download would run the
evaluation on weights nobody verified.

### GPUs

A task declares a count in `task.toml` (`[environment] gpus = 8`), never device ids. The
Docker backend turns it into `--gpus 8 --shm-size=16g` for the agent, rerun and verifier
stages (prepare only downloads and takes no cards; ray and NCCL both buffer in `/dev/shm`,
which is why the 64MB default is raised). The Harbor backend synthesizes an equivalent
compose overlay (device reservation + `shm_size`) and passes it via `--extra-docker-compose`,
because Harbor's environment declaration has no GPU field.

Host-side opt-ins, following the "explicit override" convention:

- `AUTORLBENCH_DOCKER_GPUS` — override the device spec (a 2-card machine running a task that
  declares 8);
- `AUTORLBENCH_DOCKER_SHM_SIZE` — override the shared-memory size;
- `AUTORLBENCH_DOCKER_PROXY` / `AUTORLBENCH_DOCKER_NO_PROXY` — expose a host-reachable proxy
  to the containers (Docker does not inherit the host's system proxy).

## 5. Fetch tooling travels with the task

Each task's fetch script lives in its own `environment/` directory (typically `fetch_hf.py`),
and `prepare.sh` invokes it with the image's own interpreter:

```sh
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct"
```

It downloads whole repos into the shared caches — `$AUTORLBENCH_MODELS` / `$AUTORLBENCH_DATASETS`
(defaults `<repo>/models` and `<repo>/datasets`, shared across runs and tasks) — keyed by
repo id (`owner/name` → `owner__name`), so a warm cache is a hit without re-verification. A
repo whose name would give away the task's answer lands under a neutral `--cache-as` alias
instead. The cached tree is then **hard-linked** into the workspace destination (a copy only
across filesystems), so tens of GB of weights are not duplicated on disk and later stages
read them without needing the cache at all.

Nothing of this is installed into `/usr/local` at runtime and nothing is baked into images:
a fix to a fetch script is a fix to the task that ships it.

## Environments

| Name | Upstream base (digest-pinned in the Dockerfile) |
| --- | --- |
| `areal-sglang0.5.10-torch2.9.1-flashattn2.8.3-cu129-py3.12` | `ghcr.io/areal-project/areal-runtime:v2.0.0-sglang` |
| `justrl2-data-relocation` | `ghcr.io/merak0514/autorlbench-justrl2` |
| `miles-ngc2506-distributable` | build recipe, see [Exceptions](#exceptions) |
| `miles-sglang0.5.23-torch2.14.1-flashattn2.7.4-cu130-py3.12` | `radixark/miles` |
| `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` | `slimerl/slime:v0.3.2-cu129` |
| `slime-v0.5.0rc0-sglang0.5.0rc0-torch2.8.0-flashattn2.7.4-cu126-py3.12` | `slimerl/slime:v0.5.0rc0-cu126` |
| `tool-rl-vllm024` | `python:3.12.10-slim-bookworm`, torch 2.11.0, vLLM 0.24.0 |
| `verl-qwen3vl-cu130` | `verlai/verl:vllm024.dev2` |
| `verl-v061-grpo` | legacy recipe, see [Exceptions](#exceptions) |
| `verl-0.9-vllm0.28` | build recipe, see [Exceptions](#exceptions) |
| `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` | `verlai/verl:vllm017.latest` |
| `verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10` | `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2` |
| `verl_checkpoint_host_memory_oom` | `hiyouga/verl` |
| `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` | `ghcr.io/yhyayi/autorlbench:vllm0.11-torch2.8-cu128` |
| `vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10` | `ghcr.io/yhyayi/autorlbench:vllm0.6.3-torch2.4.0-flashattn2.6.3-cu121-py3.10` |

Which tasks use which environment is declared per task in `tasks/<domain>/<task_id>/task.toml`.
