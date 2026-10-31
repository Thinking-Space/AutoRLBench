# Missing difficult examples inflate training reward

An explicitly injected stale file mapping removes difficult math examples from
JustRL2 PPO while training continues: file-read aborts rise to 50% and reward
among admitted samples rises from 0.40625 to 0.703125.

| | |
| --- | --- |
| Model | Public `openbmb/JustRL-II-base-model`, MiniCPM 2.6B class |
| Data | 64 training + 16 held-out questions, exact public UltraData Math matches |
| Training | Native JustRL2 PPO; real SGLang generation, math grading, actor and critic updates |
| Runtime | Pinned public image; 3 H100 per arm, actor1 + critic1 + generation1 |
| Recorded runs | One seed per arm, eight updates; relocation before update 4 (zero-based) |
| Contribution status | Reproducible construction evidence; companion task under `tasks/data/justrl2_data_relocation` |

## Difficulty

The mapping is in the data-loading code, not a task the model learns. The loader
holds sample IDs and file paths and opens each backing file immediately before
generation. Halfway through training, an external storage migration moves
`live/hard` to `archive/hard`. The defective manifest remains unchanged.
`FileNotFoundError` becomes a native ABORTED sample; `check_no_aborted` rejects
that group and the collector samples replacements until the batch is full.

All admitted trajectories therefore come from the easy stratum. Rewards are
still calculated from actual generated answers, and all eight actor/critic
optimizer steps succeed. A conventional reward plot looks encouraging while
the attempted training population has changed.

| Arm | Reward before | Reward after | Abort after | Admitted hard share after |
| --- | ---: | ---: | ---: | ---: |
| No relocation | 0.375 | 0.546875 | 0 | 0.5 |
| Stale mapping | 0.40625 | 0.703125 | 0.5 | 0 |
| Updated mapping | 0.390625 | 0.53125 | 0 | 0.5 |

The faulty reward increase exceeds the no-relocation increase by 0.125 and the
repaired increase by 0.15625. Its reward per attempted read falls from 0.40625 to
0.3515625. Fixed evaluation goes from 0.5 initially to 0.3125 finally; the other
arms also decline. This eight-update, one-seed case establishes the data-loss
and selection mechanism, not a significant learning advantage for the repair.
The 16-question held-out set is small, and many generations hit the token budget.

## Reference solution

Retain the storage migration and atomically publish the new hard-file mapping.
Keep the same IDs, question/answer bytes, reward rule and intended mixture.
Monitoring persistent data-access failures is also useful, but suppressing an
abort counter or changing a displayed mean does not restore hard coverage.

`runtime/` is the exact plugin used by the three recorded runs. The public data
preparer downloads four pinned UltraData Math shards and reconstructs all 64
backing files, the manifest and eval JSONL byte for byte. The original difficulty
assignments and historical accuracy values are frozen author-supplied metadata;
they are not presented as fields available in the public corpus. `CASE.json`
is intentionally regenerated with honest public-source provenance.

Run inside the pinned image, with three visible GPUs:

```bash
python3 prepare_assets.py --out /work/prepared --download
CUDA_VISIBLE_DEVICES=0,1,2 bash run_case.sh \
  --prepared /work/prepared --output /work/healthy --arm healthy
CUDA_VISIBLE_DEVICES=0,1,2 bash run_case.sh \
  --prepared /work/prepared --output /work/faulty --arm faulty \
  --reuse-native /work/healthy/native
CUDA_VISIBLE_DEVICES=0,1,2 bash run_case.sh \
  --prepared /work/prepared --output /work/repaired --arm repaired \
  --reuse-native /work/healthy/native
python3 verify.py --healthy /work/healthy --faulty /work/faulty \
  --repaired /work/repaired --out /work/comparison.json
```

Use a new output directory for every attempt. The first arm converts the public
HF weights to native checkpoint format; the later arms reuse a verified copy
of that same conversion. Private dataset paths, credentials and original training
checkpoints are not required. Public asset revisions and hashes are in
`assets.json` and `public_data/sources.lock.json`.

## Verification

`evidence/` carries all step metrics, read/admission/terminal records, actual
optimizer observations, checkpoint inventories and sanitized full stdout for
the three runs. The recorded job IDs are 1428042, 1428043 and 1428044. Hashes in
`evidence/sha256.json` bind these exported files; `sanitization.json` records path
redactions. Numeric metrics were not changed. The standalone `report.html`
summarizes the same measurements.

The complete audit inputs are also shipped in `evidence/recorded-audit.tar.gz`.
Recompute the result without trusting the summary flag:

```bash
python3 check_recorded.py --out /work/recomputed.json
# Or extract and invoke the same verifier explicitly:
mkdir -p /work/audit
tar -xzf evidence/recorded-audit.tar.gz -C /work/audit
python3 verify.py --healthy /work/audit/recorded/healthy \
  --faulty /work/audit/recorded/faulty --repaired /work/audit/recorded/repaired \
  --out /work/audit/recomputed.json
```

The archive includes structured evaluations, complete events, metadata and the
small backing dataset. Private source-path labels were redacted; transformed
CASE digests and their original digests are explicitly bound in
`recorded/transformations.json`. Replay produces the same effect sizes. It
contains no model or optimizer tensor payloads.

Use Python 3.12 or newer for the standalone preparation and archive-check commands.

`verify.py` independently recomputes the metrics, checks exact attempted/admitted
identities and terminal accounting, requires successful actor and critic updates
at every step, checks the unchanged original native checkpoint and final
optimizer checkpoint inventories, and requires the same source/data/configuration
across arms. It rejects nonpositive reward effects, missing control separation,
changed/deleted hard IDs, suppressed aborts, absent migration, and incomplete runs.
The original full audit rehashed backing files locally. Binary checkpoint files
remain on the cluster; published inventories are not a checkpoint restore test.

The generic diagnosis instruction and judge are copied verbatim from the
canonical issue scaffold. `evaluations/issue/oracle.md` is reviewer-only material.
This proposal keeps the transparent construction harness and all three arms
together. The companion task at `../../tasks/data/justrl2_data_relocation/` stages
only its defective workspace, with repair/control material under hidden
evaluations. No leaderboard score is claimed.

The separate portable-entry GPU check succeeded as job 1428553 and is recorded
in `public_runtime_validation.json`: public data preparation, exact CPU tensor
re-encoding, fresh one-GPU native conversion, and four real repaired-arm updates
on three GPUs. It used a 4096-token response budget; it verifies portability,
not a second full causal comparison. Output checkpoint payloads were rehashed
on the cluster and the original native input stayed unchanged. No checkpoint
restore or leaderboard evaluation is claimed.
