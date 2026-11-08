# PPO critic value head randomly initialised instead of zeroed

A Multi-Turn PPO reproduction on `slime` v0.3.2: Qwen3-1.7B learning to drive a
terminal on Endless Terminals tasks. The critic's value head is built with
**random** weights instead of zeros, so the critic opens with large arbitrary
value predictions it never learned, and the value loss reported throughout the
critic warmup window is inflated by roughly 77× under the calibration protocol
(~28× under the harness's short rerun) — most visibly after a reload,
where it reads like checkpoint corruption.

| | |
| --- | --- |
| **Author** | Zixi Chen |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Agentic RL |
| **Tags** | `value-head-init` `checkpoint-reload` `advantage-corruption` |
| **Framework** | slime 0.3.2 (Multi-Turn PPO), Megatron train + SGLang rollout + Ray |
| **Run time** | ~1.7 hours per run |
| **Resources** | 2 GPU |

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation. The agent sees only `workspace/` and the one evaluation's
`instruction.md`; the fixed side, the oracle, and the measured runs all live
under `evaluations/`, which it cannot read.

```
                         Stage B, first critic optimizer step
  workspace (as shipped) train/critic-value_loss  =  7.200569
  workspace + oracle fix train/critic-value_loss  =  0.093750
                                                     ^ bounded by the reward scale
```

It is not only a reporting problem. PPO optimises against the advantages that
critic produces, so over an equal 30-rollout budget the broken arm's held-out
pass rate *falls* from 0.078 to 0.043 while the repaired arm's rises from 0.080
to 0.169 — a paired difference of +0.126 over 300 held-out tasks, permutation
*p* 5e-05.

## Difficulty

`slime/backends/megatron_utils/model_provider.py`, in `LinearForLastLayer`:

```python
init_method_std = getattr(config, "init_method_std", None)
if init_method_std is None:
    init_method_std = 0.02
self.weight.data.normal_(mean=0.0, std=init_method_std)   # <-- the bug
```

`LinearForLastLayer` is what slime installs as the critic's output layer, and
it does so unconditionally — the critic branch of the model provider overwrites
whatever `output_layer` the backbone came with:

```python
model.output_layer = LinearForLastLayer(input_size=args.hidden_size, output_size=1, config=config)
```

so this constructor is the single point of control over the value head's
initial weights. `init_method_std` is the *backbone's* init scale (0.02 for
Qwen3): a reasonable default for a `[vocab, hidden]` LM head, and the wrong
idea entirely for a `[1, hidden]` value head, which should start at exactly
zero because the critic has not seen a single reward yet.

### Why the reload is what exposes it

The stale predictions do not come **out** of the checkpoint — they are created
on the way back **in**. A saved actor checkpoint holds `output_layer.weight`
with shape `[vocab_size, hidden_size]` and no `[1, hidden_size]` value head at
all, because the actor never had one. So on reload:

1. the model provider builds the backbone and replaces `output_layer` with a
   fresh, randomly initialised `LinearForLastLayer(hidden, 1)`;
2. `load_checkpoint` restores the backbone, and has nothing to restore into the
   value head, so that random weight survives untouched.

Every restart therefore opens with a different arbitrary value function. It
reads like "the critic forgot everything and is relearning", which sends you
looking at checkpoint and optimizer-state code — while the actual defect is one
line in a constructor, equally present on a from-scratch run.

### Why the number is ~7 and not something else

`V(s)` is a single linear projection of the final hidden state:

| quantity | value |
|---|---|
| `||w||₂` for `w ~ N(0, 0.02²)` over 2048 inputs | `0.02·√2048 ≈ 0.91` |
| Qwen3-1.7B final-norm hidden state, per-component RMS | `≈ 2.55` |
| ⇒ `std(V) ≈ ||w||₂ · 2.55` | `≈ 2.3` |
| ⇒ `E[V²]` | `≈ 5.3` |

slime's `value_loss_function` is the PPO clipped value loss
`max((V_clipped − R)², (V − R)²)` with **no** 0.5 factor. On the critic's first
step `values == old_values`, so clipping is inactive and the loss is exactly
`mean((V − R)²)`. With `--gamma 1.0 --lambd 1.0` the GAE return for every
response token collapses to the episode return `R ∈ {0, 1}`, and `V` is
independent of `R`, so

```
mean((V − R)²) = E[V²] − 2·E[V]·E[R] + E[R²] ≈ E[V²] + pass_rate ≈ 5.3 + 0.09
```

Measured: 7.20 on Stage B's first step, 5.32–8.15 across the warmup (the
upper end from the harness's short rerun protocol). The
estimate is low by ~25% because the hidden-state RMS is itself a rough figure
and varies by token position, but the scale it predicts is the scale observed —
what matters here is that it is set by the backbone's activation norm, a
quantity that has nothing to do with the task's rewards.

A correctly zero-initialised critic reports `E[R²] = pass_rate ≤ 1` instead —
bounded by the reward scale. The two regimes differ by ~77× with no overlap
(~28× on the harness's short 2-rollout rerun: 6.97 vs 0.25, still no overlap),
which is why the decider threshold is **1.0**: an order of magnitude from
either side.

## Reference solution

- **issue**: a passing diagnosis names the value head, its non-zero
  initialisation at construction, and does not blame the checkpoint. The
  symptom reads like "the critic forgot everything and is relearning" after a
  reload, which sends you looking at checkpoint and optimizer-state code —
  that attribution must fail: the defect is one line in a constructor,
  equally present on a from-scratch run.
- **solution**: `self.weight.data.zero_()`. One line; `LinearForLastLayer` is
  only ever instantiated for the critic with `output_size=1`, so zeroing is
  both minimal and unconditionally correct, and the actor's LM head is
  untouched.

```diff
-        init_method_std = getattr(config, "init_method_std", None)
-        if init_method_std is None:
-            init_method_std = 0.02
-        self.weight.data.normal_(mean=0.0, std=init_method_std)
+        self.weight.data.zero_()
         if bias:
             self.bias.data.zero_()
```

The production incident was closed with four layers of defence: zero at
construction, zero again after `load_checkpoint`, `optimizer.reload_model_params()`
to resync the fp32 master weights, and an offline checker. **Only the first is
the Oracle here**, and it is on its own sufficient — which took arranging,
because slime v0.3.2 randomises the value head in *two* places:

| | where | when it runs |
|---|---|---|
| (a) | `LinearForLastLayer.__init__` | always, at construction |
| (b) | `model.py:_reinitialize_critic_output_layer` | only if `_critic_output_layer_needs_reinit` |

Path (b) requires the critic's load path to be a **Megatron** checkpoint (it
looks for a `.metadata` file) whose `output_layer` shape does not match
`[1, hidden_size]`. Stage B therefore loads the critic from the HuggingFace
weights: `needs_reinit` is false, (b) never runs, and the constructor is the
only thing that sets the head, so zeroing it is the whole fix. Pointing the
critic at a Megatron checkpoint instead would also need layer 2, which is
outside the Oracle scope the case was specified with.

Layer 3 is already covered on this path — `_load_checkpoint_hf` calls
`optimizer.reload_model_params()` itself — and the constructor runs before the
optimizer exists, so there is no stale fp32 master copy of the head to resync.
Layer 4, the offline checker, is part of the solution verifier
(`evaluations/solution/verifier/tests/check_value_head.py`), not the
workspace: it used to ship inside the workspace as `tools/check_value_head.py`,
but a workspace-shipped checker that names the defect and explains its
mechanism hands the issue-level agent the diagnosis, so it lives on the
verifier side, where the agent cannot read it.

### One shared change that is not the fix

Both workspaces build the value head with `bias=False`, where upstream defaults
to `True`. Upstream slime v0.3.2 cannot load a critic from a HuggingFace
checkpoint at all with a bias: the HF loader walks every Megatron parameter and
asks the model's mapper for a source tensor, and while `output_layer.weight` is
special-cased (a `[1, hidden]` head is skipped when the HF tensor is
`[vocab, hidden]`), `output_layer.bias` has no HF counterpart and no special
case, so `qwen_hf_tensor` raises `KeyError: "Unsupported Megatron parameter
'output_layer.bias'"` before training starts. A value head does not need a
bias, and dropping it is the smallest change that lets a critic start from HF
weights. It is identical on both sides, so it cannot affect the comparison.

## Layout

```
task.toml                          task card: identity + environment + outcome
README.md                          this file (maintainers only)
environment/prepare.sh             stages Qwen3-1.7B + the prebuilt Endless Terminals prompt set
environment/fetch_hf.py            shared HF fetch helper (cache-first, hardlink)
environment/et_prompts.tar.gz      the prebuilt prompt payload (1489 admitted tasks + fixed splits)
workspace/                         the defective side the agent receives; triggers the symptom
workspace/logs/                    frozen logs + value-loss summaries of the recorded defective run
evaluations/issue/                 diagnose the snapshot  (shared protocol: instruction, oracle, verifier)
evaluations/solution/              diagnose, repair if unhealthy (instruction, oracle, verifier)
evaluations/solution/verifier/resources/run_spec.json    the harness rerun: `bash run.sh`,
                                   OUT_DIR/CKPT_ROOT under $AUTORLBENCH_RESULTS
evaluations/solution/verifier/tests/check_value_head.py   the verifier's offline zero-init check
evaluations/solution/oracle/value_head_zero_init.patch   the one-hunk fix (+ apply.sh)
evaluations/solution/oracle/measured_runs/{defective,fixed}/   the measured runs behind
                                   the Results section below, including every held-out episode
```

Only the defective side is published. The fix is a single hunk in the bundled
slime tree, `slime/slime/backends/megatron_utils/model_provider.py`; applying
`evaluations/solution/oracle/value_head_zero_init.patch` to `workspace/` yields
the control side the measured runs were produced from:

```console
$ patch -p1 -d workspace --fuzz=0 < evaluations/solution/oracle/value_head_zero_init.patch
# one hunk, in slime/slime/backends/megatron_utils/model_provider.py
```

### Inside the workspace

```
run.sh                       the two-stage reproduction (see below)
configs/model_args.sh        Qwen3-1.7B Megatron architecture flags
logs/                        frozen Stage A/B logs + value-loss summaries of the
                             recorded defective run — the symptom the agent is
                             asked to diagnose (same bytes as
                             measured_runs/defective/, unpacked)
slime/                       the slime v0.3.2 source (commit a3f50097), pruned to
                             the runtime code — the slime/ package, train.py and
                             LICENSE; docs, tests, examples, docker and CI are
                             stripped; run.sh runs this copy via PYTHONPATH
slime/slime/backends/megatron_utils/model_provider.py  THE FIX SITE
et/build_dataset.py          Endless Terminals -> slime prompt set, with validation
et/make_splits.py            deterministic train / val / test split of the admitted tasks
et/sandbox.py                per-episode terminal sandbox
et/rollout.py                multi-turn rollout + reward function (slime hooks)
et/eval_log.py               per-episode held-out reward dump (slime eval hook)
tools/parse_value_loss.py    pull critic metrics out of a slime log
tools/check_abort_path.py    offline check: can one aborted episode kill a rollout?
```

The offline value-head checker is deliberately **not** in the workspace — it
lives in the solution verifier (see "The fix" above): a workspace-shipped
checker that prints the causal story would hand the issue-level agent the
diagnosis.

(`compare_arms.py`, the paired held-out comparison of the two recorded runs,
lives with the runs it reads under
`evaluations/solution/oracle/measured_runs/`, not in the workspace.)

## Running it

Under the harness, `environment/prepare.sh` fetches Qwen3-1.7B into the
workspace (`workspace/Qwen3-1.7B`, which `run.sh`'s `MODEL_DIR` default reads)
and unpacks the prebuilt Endless Terminals prompt payload into
`workspace/et_prompts`, from which `run.sh` seeds its `ET_DATA_ROOT` — the
~40-minute transpile+validate pass is skipped, and `ET_SRC` is only read when
that payload is absent. The agent then runs `./run.sh` from `/workspace` on the
`slime-v0.3.2` image. The
short decider-only reproduction is `EVAL_INTERVAL=0`;
the full score curve below is `NUM_ROLLOUTS_B=30 EVAL_INTERVAL=5`.

The two recorded runs were produced directly, pointing `run.sh` at local mounts
(both paths are still overridable by env var, so this reproduces what is in
`evaluations/solution/oracle/measured_runs/`):

```bash
docker run --rm --gpus '"device=0"' --shm-size=32g \
  -v $PWD/workspace:/workspace \
  -v /path/to/Qwen3-1.7B:/models/Qwen3-1.7B:ro \
  -v /path/to/endless-terminals:/datasets/endless-terminals:ro \
  -v $PWD/out:/out -v $PWD/ckpt:/ckpt -v $PWD/etdata:/etdata \
  -e MODEL_DIR=/models/Qwen3-1.7B -e ET_SRC=/datasets/endless-terminals \
  -e OUT_DIR=/out -e CKPT_ROOT=/ckpt -e ET_DATA_ROOT=/etdata \
  -e NUM_GPUS=1 -e TP_SIZE=1 -e SGLANG_MEM_FRACTION=0.40 \
  -e NUM_ROLLOUTS=2 -e NUM_ROLLOUTS_B=30 -e NUM_CRITIC_ONLY_STEPS=3 \
  -e ROLLOUT_BATCH_SIZE=64 -e N_SAMPLES_PER_PROMPT=2 -e GLOBAL_BATCH_SIZE=64 \
  -e LR=2e-6 -e EVAL_INTERVAL=5 \
  -w /workspace slimerl/slime:v0.3.2-cu129 bash /workspace/run.sh
```

One 80 GB GPU is enough (`NUM_GPUS=1 TP_SIZE=1`, SGLang capped at 40% of the
device so Megatron's weights, gradients and Adam state fit in the rest); set
`NUM_GPUS=2 TP_SIZE=2 SGLANG_MEM_FRACTION=0.55` to use two (the metadata
declares 2 GPUs, matching `run.sh`'s defaults). No network at any point.

The first run validates all 2492 Endless Terminals tasks — a one-off CPU pass
(~40 min on 100 cores) written to `ET_DATA_ROOT` and reused by every later run.
After that, on one H800:

| | wall clock |
|---|---|
| Stage A (2 rollouts) | ~10 min, of which ~6 is startup |
| Stage B, per training rollout | ~94 s |
| Stage B, per eval point (150×2 + 300×4 = 1500 episodes) | ~6 min |
| **Stage B at 30 rollouts, evaluated every 5** | **~90 min** |

`run.sh` is a two-stage script:

- **Step 0-2** point `SLIME_HOME` at the bundled `slime/` tree (put ahead of the
  image's install on `PYTHONPATH`), then build, validate and split the prompt set
  (cached in `ET_DATA_ROOT`).
- **Stage A** — "the previous training run": PPO from the HuggingFace weights,
  saving a Megatron checkpoint to `${CKPT_ROOT}/actor`.
- **Stage B** — "the model was reloaded": load that checkpoint, run
  `NUM_CRITIC_ONLY_STEPS` critic-only warmup rollouts and then joint PPO to
  convergence. The warmup window's `critic-value_loss` is the defect
  measurement; the held-out score at the last eval point is the end-to-end one.

Set `SKIP_STAGE_A=1` to reuse a checkpoint an earlier invocation already wrote
— useful for restarting the long Stage B. It refuses to be silently wrong: with
no checkpoint present, Stage A runs anyway.

Artifacts in `${OUT_DIR}`: `stage_{a,b}.log`, `stage_{a,b}_value_loss.json`,
`eval_episodes/eval_rollout_NNNN.jsonl` (one row per held-out episode),
`eval_datasets.yaml`, `*_worker_logs/`, the two generated per-role Megatron
YAMLs, `splits.json`, and the dataset validation report.

`run.sh` deliberately does **not** run `pkill`/`ray stop --force` on entry, the
way slime's own example scripts do — it is written to be safe to run on a
shared host.

### The dataset

Endless Terminals ships one Docker image per task (`FROM ubuntu:22.04` + apt +
`pip install pytest`). Building ~50 images is impossible without a package
mirror and would defeat the one-image constraint, so `et/build_dataset.py`
transpiles each task's Dockerfile into a `setup.sh` whose install commands are
shimmed to no-ops, and each episode gets a private directory tree standing in
for `/home/user` (no mount namespace, no privileges — it runs on a shared host).

A task is admitted only if all four hold: setup exits 0; the task's own
`test_initial_state.py` passes; `test_final_state.py` *fails* on that initial
state; and the reference `solution/solve.sh` makes it pass. Of the repository's
**2492 generated tasks, 1489 are admitted**. The 1003 rejections are not
arbitrary — they are what the validator exists to catch:

| rejected | why |
|---:|---|
| 613 | the task writes outside `/home/user` (`/etc`, `/.bashrc`, `/dev/zero`, …), which the per-episode directory tree cannot stand in for without a mount namespace |
| 171 | the reference `solve.sh` does not make `test_final_state.py` pass in this image — an unreproducible task, scored 0 no matter what the policy does |
| 126 | the setup script fails (needs a package index, needs the `sqlite3` CLI, …) |
| 85 | `test_initial_state.py` fails, i.e. the starting state is already not what the task assumes |
| 8 | `test_final_state.py` *already passes* before the policy acts — free reward |

The report is written to `${OUT_DIR}/validation_report.json`.

The admitted set is then split by `et/make_splits.py` into **train 1039 / val
150 / test 300**, as a pure function of the admitted list and a fixed seed. Both
workspaces therefore train on and are scored on exactly the same tasks;
otherwise a comparison between them would be partly measuring which tasks each
arm happened to draw. `val` is evaluated at every eval point with 2 samples per
task and decides *when* training has converged; `test` is scored with 4 samples
per task and carries the end-to-end claim. Nothing about either run is
conditioned on `test`.

Two notes that cost real debugging time and are encoded in the workspace:

- **`/dev/shm` is mounted `noexec` in the slime image.** Many tasks assert
  `os.access(path, os.X_OK)` on a script they created, which is False there
  even at mode 755, even as root. Every scratch path defaults to `/tmp`, and
  `et/sandbox.py:check_exec_capable` fails loudly up front rather than letting
  good tasks get silently rejected.
- **`subprocess.run(timeout=)` only kills the direct child.** A task's venv
  `pip install` against an unreachable index outlived the timeout that killed
  its parent and wedged the build. Episodes and the validator both spawn with
  `start_new_session=True` and kill the whole process group, and `PIP_NO_INDEX`
  et al. make any pip fail in under a second.
- **One episode must never be able to end a training run.** `asyncio`'s
  `StreamReader` will not hand over a line until the whole line is buffered and
  raises `ValueError` past its limit — 64 KiB by default, which a terminal task
  reaches just by `cat`-ing a minified file or a base64 blob. Both of slime's
  drivers await every episode through `asyncio.as_completed`, so that one
  exception discarded the entire rollout, including the several hundred episodes
  that had already succeeded. `et/sandbox.py` raises the limit to 4 MiB, falls
  back to a bounded `read` when even that is passed, re-splits the stream into
  lines itself, and caps what it accumulates; `Episode.exec` and
  `et.rollout.generate` each additionally catch anything that still escapes and
  turn it into an ordinary failed observation / an aborted sample with reward
  0.0. A `None` reward would be just as fatal — slime's eval metrics take a
  plain `sum(rewards) / len(rewards)` over the whole split — so every path out
  of `generate` sets one.
- **…and must never be able to stall one either.** `as_completed` waits for
  every episode, so one that simply never finishes is worse than one that
  raises: there is no traceback and no exit code, just a run that stops making
  progress. Both halves of that happened here. One Endless Terminals task
  instructs the policy to leave `sleep 1000` and `yes 'Compiling docs...'`
  running in the background; asyncio resolves `Process.wait()` only once the
  child has exited *and* every pipe it was handed is at EOF, so a background job
  holding the episode shell's stdout keeps that future pending after the shell
  is dead. Separately, slime builds its HTTP client with `httpx.Timeout(None)`,
  so a `/generate` that never comes back waits forever. An eval sat at 1199 of
  1200 episodes for half an hour. `et/sandbox.py` therefore reaps killed
  processes under a 10 s bound instead of unconditionally, and `et.rollout`
  caps the whole episode — generation, shell, verifier, teardown — at
  `ET_EPISODE_TIMEOUT` (900 s), far above any legitimate episode, so it only
  ever fires on a hang.
- **An episode that gives up must still look like a sample.** slime decides
  whether an optional per-sample field exists by inspecting `samples[0]` alone
  and then indexes *every* sample in the batch — `metadata["round_number"]` at
  `ray/rollout.py:382`, `rollout_log_probs` at `:386`. So the batch that breaks
  is the ordinary one: a normal episode first, and behind it a straggler that
  SGLang aborted at the end of the rollout. The first gap cost a rollout to
  `KeyError: 'round_number'`; fixing only that exposed the second as
  `TypeError: must be real number, not NoneType` inside `torch.as_tensor`,
  because an episode aborted before its first token has no log probs at all.
  Both tracebacks name slime, not this workspace, and both took an hour of
  training to reach. `et.rollout._abort` is now the single exit for every
  giving-up path and fills all three indexed fields, giving a
  nothing-generated episode one *non-trainable* end-of-text token so its
  token, mask and log-prob lists are present, equal-length and zero-weighted
  instead of absent. `tools/check_abort_path.py` reproduces the mixed batch
  offline in seconds, with no GPU and no model.

## Results

One recorded run per side (defective and control) on 1×H800 with the command
above — the threshold is calibrated on these, one run per side, so the margins
below are reported without a variance estimate: Stage A = 2
rollouts of joint PPO, Stage B = 30 rollouts of which the first 3 are
critic-only, 64 prompts × 2 samples = 128 episodes per rollout, global batch 64
⇒ **2 critic optimizer steps per rollout**, so the warmup window is the first
**6** critic steps. Held out: `et_val` (150 tasks × 2 samples) at every eval
point, `et_test` (300 tasks × 4 samples) carrying the end-to-end claim. Raw
artifacts are in
[evaluations/solution/oracle/measured_runs/](evaluations/solution/oracle/measured_runs/).

### Stage B warmup — the decider

`train/critic-value_loss` over the critic-only window:

| critic step | `defective` | `fixed` |
|---|---|---|
| **0**  ← decider | **7.200569** | **0.093750** |
| 1 | 7.645938 | 0.139836 |
| 2 | 6.588119 | 0.030926 |
| 3 | 6.327897 | 0.015477 |
| 4 | 5.419277 | 0.061350 |
| 5 | 5.315719 | 0.152233 |
| mean over the window | 6.416253 | 0.082262 |
| max over the window | 7.645938 | 0.152233 |
| rollouts generated | 30 | 30 |
| mean response length, warmup rollouts | 173.9 tok | 238.6 tok |
| verdict vs threshold 1.0 | `> 1.0` — symptom present | `< 1.0` — inside the reward scale |

The repaired arm's opening number is not merely "small", it is the value the
derivation predicts exactly. With `V ≡ 0` and `γ = λ = 1` the clipped value
loss on the first micro-batch is `mean(R²) = mean(R)`, the fraction of those 64
episodes that passed: **0.093750 = 6/64**. Only step 0 is exact — one optimizer
step later the head is no longer zero and the identity stops holding.

The broken arm shows no such structure. Its loss opens at `E[V²] + mean(R)`,
where `V` is a random projection of the backbone's final hidden state, so the
number is set by the activation norm of Qwen3-1.7B and has nothing to do with
how the episodes went. That is the whole complaint: during warmup the metric is
reporting the initialisation, not the learning.

### Held-out score — the end-to-end claim

Episode pass rate on `et_test`, 300 tasks × 4 samples = 1200 episodes per point
(the repaired arm has no point at rollout 4: that eval stalled at 1199 of 1200
episodes — the hang that `ET_EPISODE_TIMEOUT` now bounds — and the run was
killed there and resumed from rollout 5, so no dump was written):

| eval @ rollout | `defective` | `fixed` |
|---:|---|---|
| 0 | 0.0783 | 0.0800 |
| 4 | 0.0192 | — |
| 9 | 0.0525 | 0.1108 |
| 14 | 0.0192 | 0.1325 |
| 19 | 0.0292 | 0.1242 |
| 24 | 0.0517 | 0.1433 |
| **29** | **0.0433** | **0.1692** |

The two arms are indistinguishable where they start and far apart where they
end. Paired per task, over the same 300 tasks:

| | at rollout 0 | at rollout 29 |
|---|---|---|
| paired mean difference (fixed − buggy) | +0.0017 | **+0.1258** |
| 95% CI | −0.0122 … +0.0155 | +0.0937 … +0.1579 |
| sign-flip permutation *p* (20000 draws) | — | **5e-05** |
| McNemar on "solved at least once" | — | buggy-only 1, fixed-only 56, *p* 8.0e-16 |

`et_val` says the same thing with a quarter of the episodes: 0.0400 vs 0.0967
at rollout 29, paired difference +0.0567 (95% CI +0.0107 … +0.1026),
permutation *p* 0.024. Both verdicts are reproducible from the archived
per-episode dumps:

```bash
cd evaluations/solution/oracle/measured_runs
python3 compare_arms.py \
    --buggy defective/eval_episodes --fixed fixed/eval_episodes \
    --dataset et_test
```

So the defect is not cosmetic. A critic whose baseline is a random projection
produces advantages that are mostly a function of the prompt rather than of
what the policy did, and PPO optimises against them faithfully: the broken
arm's training reward falls from 0.0328 (rollouts 3-7) to 0.0250 (rollouts
25-29) while the repaired arm's rises from 0.0625 to 0.1000, and its mean
response length collapses from 174 to 93 tokens as the policy learns to stop
trying. It ends below where it started — the reload did not merely inflate a
number, it cost the run.

### What the broken critic does after the warmup

Over the full 30 rollouts the broken arm's value loss does decay — 7.20 at step
0, 1.88 by step 10, a minimum of 0.0886 by the end — because the critic is
perfectly able to learn its way out of a bad initialisation. That is exactly
why the symptom is reported as a *warmup* problem and why the decider is the
first step: by the time the number looks normal, the actor has already been
trained against it. Averaging the value loss over all 60 steps gives 1.54,
which is why `run.sh` passes `--warmup-steps` derived from
`NUM_CRITIC_ONLY_STEPS` rather than letting the window grow with the stage.

### Stage A — the same defect before any reload

Stage A trains actor and critic together from the HuggingFace weights. It is
not the decider, but it is worth recording, because it shows the inflation is
not about reloading at all — a from-scratch run has it too. The reload is only
what makes it *look* like checkpoint corruption.

| critic step | `defective` | `fixed` |
|---|---|---|
| 0 | 6.887251 | 0.109375 |
| 1 | 7.034253 | 0.139836 |
| 2 | 6.531444 | 0.015483 |
| 3 | 6.225787 | 0.015431 |
| mean | 6.669684 | 0.070031 |
| mean response length | 217.0 tok | 216.1 tok |

Here too the repaired arm's first step is exact: **0.109375 = 7/64**.

Stage A's own `rollout/rewards` (0.0352 vs 0.0703) is over 256 episodes total
and is not evidence of anything; the honest statement that the two arms start
from the same place is the rollout-0 row of the held-out table above, which is
1200 episodes per arm and puts the paired difference at +0.0017 with a
confidence interval straddling zero.

### On convergence

30 rollouts is an **equal fixed budget**, not a measured plateau. The repaired
arm's validation curve was still rising at the last point (0.0933 → 0.0967),
so the reported +0.1258 is a lower bound on the gap rather than a
converged-vs-converged comparison; the broken arm had clearly stopped improving
long before. Nothing about either run was conditioned on `et_test`.

## Verification

- **issue**: the shared issue protocol — an LLM judge scores `agent_answer.md`
  against the hidden `oracle.md` on verdict / root cause / evidence
  (0.2 / 0.6 / 0.2), pass requires verdict_score 1.0 and root_cause/evidence
  each above 0.8. A passing diagnosis names the value head, its non-zero
  initialisation at construction, and does not blame the checkpoint. The
  protocol (`instruction.md` + `verifier/{test.sh,judge.py,judge_prompt.md}`)
  is byte-identical to the other LLM-judge tasks; only `oracle.md` is
  task-specific. It diagnoses the workspace as a historical snapshot — the
  symptom is readable in the frozen `workspace/logs/`, the root cause is
  visible in `slime/slime/backends/megatron_utils/model_provider.py` — and
  does not launch training.
- **solution**: a deterministic metric verifier — the agent diagnoses and,
  if the run is unhealthy, repairs it before the rerun. `verifier/tests/check_value_head.py` (the verifier's own offline
  check) passes, and Stage B reports `first_value_loss < 1.0` and
  `warmup_value_loss_mean < 1.0` over ≥1 critic step, with ≥1 rollout actually
  generated: Stage B's first critic step must land inside the reward scale. It
  guards the degenerate ways to satisfy it — a run with zero critic steps,
  zero rollouts, or no generated tokens fails rather than passes, so the
  metric cannot be brought under the bound by never reporting it. It accepts
  `AUTORLBENCH_SKIP_TRAIN=1` to run the offline check alone on a GPU-less
  machine; it then exits 2 and prints `SOLUTION_PARTIAL`, which is not a pass.

The solution verifier does not gate on the held-out score, deliberately.
Settling that question costs 30 rollouts and seven eval points — about 100
GPU-minutes per arm — whereas the warmup value loss separates the two sides by
77× within the first critic optimizer step. The held-out comparison in Results
is what establishes that the defect is worth catching; the first critic step is
what makes catching it cheap. Both run from the same `run.sh`: set
`EVAL_INTERVAL=0` for the decider alone, or `NUM_ROLLOUTS_B=30 EVAL_INTERVAL=5`
to reproduce the
score curve.
