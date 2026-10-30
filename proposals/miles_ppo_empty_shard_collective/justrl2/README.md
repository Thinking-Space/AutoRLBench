# Public JustRL2 constructed regression case

**Status (October 30, 2026): the final public image passed a real 8×H100
healthy/faulty training pair and native model/Adam checkpoint restoration.** A
separate CPU task reran the acceptance checks against the preserved artifacts
and rehashed the original and newly saved checkpoint files. This remains a
proposal, without task registration or a leaderboard result.

The case pins public `merak0514/JustRL-II` at
`75814318f243af7e39af337c1173fc3927c9635b` and public
`openbmb/JustRL-II-base-model` at
`b558bbc71d9433e7c5869effcbac07d4c7f2c5b4`.
**The pinned public source already fixes the empty-shard defect.** The faulty
arm deliberately reintroduces its local guard; it is not an unfixed upstream bug.

## Validated image and evidence

```text
ghcr.io/merak0514/autorlbench-justrl2@sha256:8d2eb1fe292a4e381915734b7854213002ca9bae6b50efab1bd8cc75cad14a8b
```

[image_build.json](image_build.json) records the successful linux/amd64 build,
manifest/config hashes, anonymous registry access and authenticated Ray build
smoke. The same image digest was used for all three GPU stages. No packages
were modified during those runs. The frozen tooling bundle SHA256 is
`79907c3176a798886bbe7c237f8622c413bd031894423dd56dc0cccbc03d1f61`;
the [final receipt](public_runtime_results.json) also pins each runtime script.
The compatibility helper is preparation-time source instrumentation, not an
additional image layer.

| Stage | Verified result |
| --- | --- |
| Healthy | Actor4 + critic4 complete real forward, GAE, whitening, backward and finite Adam updates; both roles save checkpoint 0; training returns 0. |
| Faulty | In both roles rank 1 skips whitening on its empty shard while rank 3 enters; the target `all_reduce` fails with the complete `DistBackendError` / `store->get` / `wait timeout after 60000ms` stack; training exits 1. |
| Restore | All eight ranks load exact native model and populated Adam state, advance step 1 to step 2, save checkpoint 1, and verify its payload against live post-update state; training returns 0. |
| Independent audit | A separate 1-CPU, 0-GPU task reruns healthy/faulty and restore event checks, reads both generations' DCP inventories and rehashes every referenced payload; original checkpoints are unchanged. |

All stages used eight distinct physical H100 GPUs across eight single-GPU
process nodes, each with 4 CPUs and 96 GiB host memory. Each role retains
TP1/PP1/CP2/DP2. No stage reached its outer deadline. Native training completion
is checked separately from the cluster wrapper's success status; the faulty
wrapper succeeds only when the expected training failure passes verification.

- [public_runtime_results.json](public_runtime_results.json): input identities,
  return codes, verification results, eight-rank model/Adam snapshots, log hashes,
  physical-device counts and independent audit provenance.
- [public_runtime_events.json](public_runtime_events.json): sanitized rank events;
  numerical training values and CP/DP membership are unchanged.
- [public_faulty_trace.txt](public_faulty_trace.txt): complete target Python
  traceback, with extraction provenance in the final receipt.
- [public_checkpoint_inventory.json](public_checkpoint_inventory.json): actor and
  critic checkpoint 0/checkpoint 1 trackers, metadata keys and referenced-file hashes.
- [runtime_checks.json](runtime_checks.json): earlier component checks and final
  image validation, distinguished by digest and scope.

Private run paths, node addresses and process IDs are removed or aliased. Raw
artifacts and checkpoint tensors are retained privately; no checkpoint tensors
are redistributed in this proposal. Earlier internal-runtime results remain in
[restore_results.json](restore_results.json) and [restore_events.json](restore_events.json),
separately from the final public-image evidence above.

## Paired source and compatibility

`build_case.py` verifies 286 public source files against `source_files.json` and
creates `healthy/` and `faulty/`. Both receive identical instrumentation and
launch helpers. Their only difference is the injected `numel()>0` guard and its
indentation in `miles/backends/training_utils/loss.py`. `defect.patch` and
`pair_manifest.json` record that difference and every file hash. The revealing
upstream fix comment is removed from both trees.

The image uses Torch 2.13+cu129, a TE 2.17 PyTorch binding rebuilt against that
runtime, Megatron 37f91617 and Ray 2.59 with matching OpenTelemetry API/SDK 1.44.
Incompatible optional FA2/FA3 packages are removed; cuDNN fused attention stays
enabled. Both arms disable optional Apex gradient-accumulation fusion while
retaining real TE Adam updates. Public model conversion verifies every HF
tensor before recording native checkpoint inventories.

The common `process_group_compat.py` helper rebinds Megatron's cached collective
callables to JustRL2's existing native-process-group forwarding, including the
renamed PyTorch tensor-collective aliases. This resolves a runtime compatibility
crash before optimizer execution. It changes no collective operation, group
membership or normalization. A two-H100 check verified exact synchronous and
asynchronous reduce-scatter/all-gather outputs after rebinding;
[process_group_compat_checks.json](process_group_compat_checks.json) records it.
The empty-shard guard remains the only healthy/faulty source difference.

## Authored replay fixture

`make_replay.py` encodes two readable, scripted math trajectories with the pinned
public tokenizer. Responses and rewards are authored, not model-generated.
Every sample has `rollout_log_probs=None`; the actor computes real logprobs
during training. No online SGLang generation or fabricated inference logprobs
are used.

| Sample | Prompt tokens | Response tokens | CP0 / CP1 response tokens | Reward |
| --- | ---: | ---: | --- | ---: |
| Long reference context, short wrong answer `3` to `1+1` | 422 | 2 | 2 / 0 | 0 |
| Short prompt, longer explanation ending in `2` | 28 | 329 | 149 / 180 | 1 |

Both responses include the end-of-message token, and both masks contain only
ones. With balancing off, GBS2/MBS1 and DP2 give one sample to each DP replica.
The CP1 coordinate's DP group therefore receives 0 versus 180 local response
tokens. This is an empty tensor, not a nonempty tensor whose mask sums to zero.
Deliberate text repetition is documented in [fixture.json](fixture.json).

Serialization uses public `Sample.to_dict()` and is round-trip checked.
The canonical readable JSON SHA256 is
`ecb7d044a6c515ab897a169e2bd352e5a0397d7b26f38beb1c6870dec4d8060c`.
`replay.pt` feeds the native `--load-debug-rollout-data` path. Public model
architecture, scalar critic head/bias 0.52 and length-adaptive GAE (`k=0.5`) are
retained. There is one update per stage, no critic-only warmup, length shaping,
TIS, balancing, offload or dynamic microbatching. This tests distributed
training behavior, not learning-curve reproduction or policy quality.

## Preparation and execution

From this directory, with Python 3.12+ inside the image, prepare fresh paths:

```bash
python prepare.py --output /workspace/prepared
python connect_frameworks.py --prepared /workspace/prepared
python convert_model.py --prepared /workspace/prepared --output /workspace/native-model

BENCH_DEDICATED_CONTAINER=1 python run_pair.py \
  --prepared /workspace/prepared \
  --native-model /workspace/native-model \
  --output /workspace/pair-run

BENCH_DEDICATED_CONTAINER=1 python run_restore.py \
  --prepared /workspace/prepared --native-model /workspace/native-model \
  --checkpoint-root /workspace/pair-run/healthy \
  --output /workspace/restore-run --timeout 2400
```

`assets.json` pins public source/model/tokenizer downloads by revision, byte size
and SHA256. Model weights are downloaded, not bundled. The launcher owns a
private Ray instance in a dedicated container/job. Dedicated external Ray also
supports two 4-GPU, four 2-GPU or eight 1-GPU process nodes, with eight distinct
physical GPUs and unchanged actor4/critic4, CP2/DP2 geometry.

## Acceptance checks and limits

`verify_run.py` requires all eight ranks' reciprocal group ledgers, independent
pooled normalization, actual finite optimizer steps and sampled FP32 parameter
changes for both roles. Healthy DCP metadata must reference model and optimizer
payloads with complete shard files. Metadata presence alone is not restoration;
that check deliberately returns `checkpoint_restore_tested=False`.

The faulty checker requires the empty peers to advance past the guard and their
nonempty peers to enter without completing whitening. It additionally requires
correlated watchdog evidence or the complete target store-timeout stack. A
generic timeout or outer deadline does not pass. The first divergent collective
comes from the per-rank ledger; a later watchdog may report another operation.

`run_restore.py` independently compares keyed full native model/Adam tensor
hashes during loading and checkpoint 1 readback. It rejects fresh initialization,
swapped moments, skipped updates and changed original checkpoints. Only storage
alignment explicitly tagged as padding by the live native template is excluded,
matching the native loader; counts are recorded. No actual model parameter or
Adam moment is excluded. All eight ranks must emit one load, one successful
update and one save in order, giving 24 restore acceptance events. The final
event file also retains the training instrumentation events.

The local tooling/restore/compatibility suite passes 66 tests, and the private
stage controller passes 25. These tests include real CPU Adam restoration,
malformed evidence rejection, and native process-group installer checks. The
older four-process CPU/Gloo source-path check used supplied critic values 0.52;
it is distinguished from real GPU training in [construction_checks.json](construction_checks.json).

```bash
python check_public_path.py --pair /workspace/prepared/pair \
  --replay /workspace/prepared/replay/replay.json --output /tmp/public-path-check
python -m pytest test_case_tools.py test_restore.py test_process_group_compat.py -q
```

## Before task registration

This directory stays under `proposals/`. Maintainers still need to agree on
scope and acceptance calibration, separate the defective-only agent workspace
from hidden repair/verifier material, add hidden numerical/adversarial cases,
and run AutoRLBench's task validator. The source-based training case, image and
native restoration are verified; this is not a calibrated hidden grader.

Scripts here are Apache-2.0; the copied source retains its LICENSE. The pinned
HF model card has no explicit license field, so model-use terms need resolution
before final task acceptance. No model weights or checkpoint tensors are bundled.
