# Public JustRL2 constructed regression case

This directory builds a training-replay candidate from the public
[`merak0514/JustRL-II`](https://github.com/merak0514/JustRL-II) repository at
`75814318f243af7e39af337c1173fc3927c9635b`. It reintroduces the local empty-shard
guard around advantage whitening. **Public HEAD already fixes that defect.**
The mutation is intentional and is not presented as an unfixed upstream bug.

**Status (October 29, 2026): executable construction draft; CPU path and one
real GPU actor/critic replay pair and native checkpoint restoration verified.** It is not an AutoRLBench task
registration or a leaderboard result. The preceding private-history and
synthetic H100 kernel results do not stand in for this candidate's training run.

## What the pair contains

`build_case.py` verifies all 286 public source files against `source_files.json`,
then creates `healthy/` and `faulty/`. Both receive identical local instrumentation
and launch helpers. The only difference between the trees is one `numel()>0`
guard and indentation in `miles/backends/training_utils/loss.py`. The generated
`defect.patch` and `pair_manifest.json` record that difference and every file hash.

The common instrumentation records role/rank/group membership, the first call
divergence, before/after whitening values, and actual optimizer return/gradient
and sampled FP32 parameter changes. It adds no collectives and does not alter
model values, advantages or gradients. It removes the answer-revealing fix
comment from both copies. These paired trees and scripts are reviewer material;
the good tree and verifier must be hidden in an eventual benchmark task.

## Constructed input, with explicit provenance

`make_replay.py` uses the pinned public JustRL2 tokenizer to encode two readable,
scripted math trajectories. Neither response is claimed to be model-generated.
No inference logprobs are fabricated: every sample has `rollout_log_probs=None`.
The actor will compute its real logprobs during replay training.

| Sample | Prompt tokens | Response tokens | CP0 / CP1 response tokens | Authored reward |
| --- | ---: | ---: | --- | ---: |
| Long reference context, short wrong answer `3` to `1+1` | 422 | 2 | 2 / 0 | 0 |
| Short prompt, longer correct explanation ending in `2` | 28 | 329 | 149 / 180 | 1 |

Text repetition in this fixture is deliberate and documented. Both responses
include the tokenizer's end-of-message token. Both masks contain only ones.
With balancing off, GBS2 and DP2 assign one sample to each DP replica. The CP1
position's DP group therefore receives **0 versus 180** local response tokens.
This is distinct from a nonempty tensor with a zero-valued mask.

The fixture is serialized through public `Sample.to_dict()` and round-trip
checked. `replay.json` is the canonical readable record; `replay.pt` is generated
for JustRL2's native `--load-debug-rollout-data` path. The canonical JSON SHA256
under the checked tokenizer is
`ecb7d044a6c515ab897a169e2bd352e5a0397d7b26f38beb1c6870dec4d8060c`.
The authored canonical record is included as [fixture.json](fixture.json).

## Real training path

The launcher uses public `train.py` with actor4 + critic4 GPUs, each TP1/PP1/CP2/DP2.
It retains real public model forward passes, actor/critic value exchange,
JustRL2 length-adaptive GAE (`k=0.5`), whitening, backward, Adam update and
actor/critic checkpoint saving. Replay uses authored trajectories and rewards;
it does not run an online SGLang rollout. This path completed in the healthy arm
of the October 28, 2026 8×H100 run. The faulty arm reached the intended
empty/nonempty divergence: the empty rank advanced past the injected guard,
the nonempty peer entered `distributed_masked_whiten`, and its
`dist.all_reduce(stats_tensor, group=process_group)` failed with
`DistBackendError` / `store->get` / `wait timeout after 60000ms`.

Task-specific settings: one rollout/update, GBS2/MBS1, no critic-only warmup,
no length shaping, no TIS (no generated logprobs), no balancing, no offload and
no dynamic microbatching. The public model architecture, scalar critic head and
bias0.52 are retained. This is a systems regression case, not a reproduction of
the published 128k learning curve or a measurement of policy quality.

`run_pair.py` runs the healthy arm first in a **dedicated container/job**. It
stops if the intact training path fails. It verifies pinned HF assets, a recorded
native conversion receipt, native checkpoint inventory, replay hash and both
source inventories before execution. The faulty arm uses the same assets and
launch settings. The helper owns a private Ray instance; the parent performs
Ray cleanup only inside the dedicated task container.

## Preparation and execution

The candidate runtime was built on October 29, 2026 (linux/amd64):

```text
ghcr.io/merak0514/autorlbench-justrl2@sha256:cc32d641e90f613029bc64e802dd330f5205b26cf3bb447e67ebb485a68ea0d2
```

[image_build.json](image_build.json) records the successful GitHub Actions build,
base digest, Dockerfile hash and output digest. Anonymous registry access and the
[manifest digest](image_manifest.json) were verified. The Dockerfile uses the
fixed public base digest `49637ea9...18164c5`. The first build failed its import smoke test because ANTLR4.13 was incompatible
with OmegaConf. The current recipe pins ANTLR4.9.3 and runs parser assertions
during construction; the replacement build passed. The receipt retains the
failed attempt separately. The second CPU-only smoke check passed the parser
imports but could not load TransformerEngine without the NVIDIA driver
(`libcuda.so.1`). It also reported a FlashAttention ABI warning. Full module
import and the configured fused-attention training path require GPU validation.

**GPU training inside this newly built image is not yet validated.** The prior
GPU replay used a different internal image. The build emitted dependency-conflict
warnings from the broad upstream environment; a successful build is not proof
of training compatibility. Native checkpoint restoration passed on the prior internal compatibility image.
It has not yet been tested in the newly built public image.

From this directory, in a suitable environment (Python3.12+ for the preparation
helpers), use a new preparation directory:

```bash
python prepare.py --output /workspace/prepared
python connect_frameworks.py --prepared /workspace/prepared
python convert_model.py --prepared /workspace/prepared --output /workspace/native-model

BENCH_DEDICATED_CONTAINER=1 python run_pair.py \
  --prepared /workspace/prepared \
  --native-model /workspace/native-model \
  --output /workspace/pair-run
```

`assets.json` pins the public source archive and each HF model/tokenizer file by
revision, size and SHA256. Model weights are downloaded, not redistributed here.
`convert_model.py` runs the public HF-to-torch_dist converter and records the
input hashes plus tracker-selected output metadata and shard hashes. The GPU
replay completed this conversion with lossless tensor checks; the standalone
candidate image still needs its own training environment validation.

## Acceptance criteria

`verify_run.py` is a **candidate evidence checker**, not yet a calibrated hidden
AutoRLBench grader. It requires:

- Both roles' actual pre-whitening ledgers establish the expected DP group
  with one empty and one nonempty local response shard.
- The healthy run completes whitening and an actual successful finite optimizer
  step on all actor/critic ranks, with sampled FP32 changes demonstrated for each
  role. Outputs match independent pooled normalization.
- Actor and critic trackers select checkpoint0; PyTorch DCP metadata contains
  model and optimizer state, and every referenced shard exists with sufficient
  bytes. **This is not a restore test**; `checkpoint_restore_tested=False` remains.
- In the faulty run, empty peers advance past the guard without calling
  whitening, nonempty peers enter the call and do not complete it, and either
  correlated watchdog lines or the complete target `DistBackendError` /
  `store->get` / `wait timeout` / `all_reduce` stack is present. A generic
  timeout or an outer deadline alone cannot pass.

Since skipped peers may advance into other collectives, the final watchdog
operation need not itself be the whitening reduction. The first divergent
operation is established from the role/rank call ledger, not guessed from one
error string. Further hidden numerical/adversarial cases and a real restore
check should be added before claiming Solution-task readiness.

## Checks performed

- Public source and tokenizer download hashes verified; tokenizer-only
  preparation executed end to end; pair has exactly one differing source file.
- `check_public_path.py` ran four real CPU/Gloo processes using the exact public
  CP, GAE and normalization functions compiled from their source AST. Healthy
  returns and pooled normalization passed; the injected guard timed out in the
  expected nonempty peer. Critic values were supplied as0.52 in this check;
  **no model forward, backward or GPU training is claimed**.
- Tooling tests exercise real CPU Adam snapshots, durable evidence writes, DCP
  metadata/shard inspection, the target store-wait verifier path, and rejection
  of malformed/missing evidence. The combined tooling and restore suite passes 61 tests.
- The October 28, 2026 GPU replay pair used the pinned public model and source
  revision. The healthy arm passed the evidence checker with one actor and one
  critic optimizer step plus model/optimizer checkpoint payloads. The faulty
  arm passed the checker with actor and critic first-divergence records and the
  target `DistBackendError` store wait. A 1-CPU read-only verifier task reran
  both checks against the preserved shared artifacts. [construction_checks.json](construction_checks.json)
  records the earlier GPU result separately from the new image build and
  the separately verified native GPU restore path.

Reproduce the CPU source-path check after preparation:

```bash
python check_public_path.py --pair /workspace/prepared/pair \
  --replay /workspace/prepared/replay/replay.json --output /tmp/public-path-check
python -m pytest test_case_tools.py -q
```

The public model conversion and native checkpoint inventory completed in the GPU
run. The healthy arm performed one real replay update for both actor and critic;
the faulty arm intentionally failed at the target collective after the first
divergence. This validates the constructed training case, not online rollout
quality or checkpoint restoration.

A subsequent CPU-only read of the saved actor and critic checkpoint0 metadata
completed successfully; [checkpoint_inventory.json](checkpoint_inventory.json)
records tensor counts and common-state keys. It does not establish successful
optimizer restoration.

## Checkpoint restore validation

`run_restore.py` validates both roles with the original healthy checkpoints as
read-only inputs and a separate output tree:

```bash
BENCH_DEDICATED_CONTAINER=1 python run_restore.py \
  --prepared /workspace/prepared --native-model /workspace/native-model \
  --checkpoint-root /workspace/pair-run/healthy \
  --output /workspace/restore-run --timeout 2400
```

It requires native loading of model and populated Adam state on every rank,
compares keyed full tensor hashes against an independent checkpoint read, runs exactly
one additional update, and checks that checkpoint1 matches the live post-update
state. It rejects fresh initialization, swapped Adam moments and optimizer-skip paths.
Native `dp_reshardable` alignment records explicitly tagged by the live template
as padding are excluded, matching the native loader; their tensor/element counts
are recorded. All model and optimizer parameter tensors remain exact comparisons.
Original
checkpoint hashes must remain unchanged. `RESTORE_RESULT.json` is written only
after all checks succeed; the original metadata checker alone remains a
payload-presence check. The focused CPU tests exercise real Adam checkpoint
restore and reject a no-op loader. Five additional checks ran with the exact
pinned Megatron mapping classes ([receipt](native_mapping_checks.json)), after
a GPU attempt exposed an inaccurate test double for `LocalNonpersistentObject`.
Its native API uses `unwrap()`, not `.data`. These CPU checks are separate from the native GPU result below.

## Native restore result (October 29, 2026)

[restore_results.json](restore_results.json) and the sanitized
[rank events](restore_events.json) record one successful restore on two nodes
with four H100 GPUs each, using the internal compatibility image. Each role
retains four ranks and CP2/DP2. All eight ranks passed exact native model/Adam
load verification, completed rollout1, advanced Adam step1 to step2, and passed
an independent readback of checkpoint1 against the live post-update state.
Original actor/critic checkpoint hashes were unchanged. The parent returned0
without reaching its outer deadline; all24 load/update/save events passed a
second offline check. No checkpoint tensors are distributed in this proposal.

The critic rank0 comparison excluded189 alignment-storage elements (63 each
for the parameter storage, first moment and second moment), exactly those the
native loader discards. No actual parameter or Adam moment was excluded.
The successful run used256GiB host memory per node; a prior128GiB attempt ran
both updates but was rejected after a post-save readback OOM. Earlier verifier
failures are not counted as successful restores. **The public candidate image
still requires its own GPU training/restore run.**

## Packaging next

After a real healthy/faulty pair is recorded, place only the defective workspace
and sanitized frozen evidence into `tasks/infrastructure/.../workspace/`, move
the repair and acceptance tests to hidden evaluation directories, add a validated
shared image and pinned preparation script, then run AutoRLBench validation.
The current directory stays under `proposals/` until those conditions hold.

All scripts here are Apache-2.0. The copied JustRL2 tree retains its own LICENSE.
The HF model card currently omits an explicit license field, so model-use terms
should be resolved before final task acceptance; no model weights are bundled.
