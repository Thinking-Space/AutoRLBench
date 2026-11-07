# Revised contestant boundary and topology checks

This revision addresses the November 6, 2026 review of PR #12. The earlier
directory is an audit bundle: it contains the answer and must **never** be the
contestant workspace. `export_challenge.py` now creates a separate, reproducible
contestant archive and a maintainer-only directory. The original GPU evidence
remains unchanged.

| Review finding | Revision |
| --- | --- |
| Traceback gives the exact function and line | Keep the original trace for reviewers; expose only the actual terminal backend exception, with the truncation explicitly declared. |
| Event names explain the fault | Project two rank-local progress checkpoints to ordinal labels. Drop exact call events, intermediate tensors and local counts; retain their source mapping outside the contestant mount. |
| `grep whitening` immediately identifies the answer | The observation view and instructions contain no whitening keyword or source pointer. Strip the upstream repair comment and all targeted diagnostic calls; keep native function names and the real competing communication paths. No dummy decoy code. |
| CP/DP reasoning is optional | Require the token slicing, concrete rank mapping, differing local decisions and fixed-size DP reduction in the diagnosis rubric. Numerical checks reject group changes, normalization bypasses and local-mask workarounds. |
| Reference repair sits beside the defect | Export no git history, control source, verifier, patch, original README or audit data into `workspace/`. Only `workspace.tar.gz` is delivered to the contestant. |

This is a **candidate hardening layer**, not a registered `tasks/` entry or a
calibrated difficulty claim. In particular, a written rubric does not by itself
enforce a benchmark's judge behavior. Integration must adopt its causal-chain
gate and test it against incomplete diagnoses before scoring submissions.

## Build the candidate

Python 3.12+ is required to parse the pinned native source. The exporter itself
uses only the standard library. Obtain the pinned public source during trusted
preparation, then run from this directory:

```bash
git clone https://github.com/merak0514/JustRL-II /tmp/justrl2-source
git -C /tmp/justrl2-source checkout --detach 75814318f243af7e39af337c1173fc3927c9635b
python export_challenge.py --source /tmp/justrl2-source --output /tmp/update-case
```

The exporter verifies the complete 286-file source inventory before selecting
native framework modules, plugins, model arguments and entrypoints. It rejects
modified archives, links, stale evidence hashes and existing output paths.
Temporary assembly is renamed into place only after successful validation.
Repeated exports have identical file hashes and archive bytes.

```text
/tmp/update-case/
  workspace.tar.gz          # only contestant delivery
  workspace/               # native framework + launch + input/observation view
  maintainer/
    manifest.json          # every original/exported hash and transformation
    repair.patch           # reference repair against the exported source
    oracle.md              # causal-chain rubric and wrong diagnoses
    topology_verifier.py   # executable native-source checks
```

Mount **only** `workspace/` at `/workspace` in a fresh, network-disabled agent
container. Do not mount this repository or the export parent. Audit the image
and other mounts for an installed repaired JustRL2 copy or cached source
history; the exporter cannot enforce container isolation. No opaque filenames,
fake tracebacks, fabricated events or renamed computational symbols are used.

The public image digest and raw pair still appear in the parent README for
reviewers. Model weights and converted checkpoints are not part of this
archive. The original recorded model/Adam run is evidence of the defect; the
exported launch still needs external assets and framework setup for GPU use.
The frozen JSON can be serialized with `torch.save(json.load(...), path)` into
the native debug replay format without regenerating any token or reward.

## Review and test

The [November 7 validation receipt](validation.json) records **104 passing
tests, zero failures and zero skips**: the existing 66 tests, 16 export tests
and 22 topology/verifier tests. A separate run against the actual exported
workspace rejected the defect with a concrete participation-mismatch witness;
the reference repair passed all 36 topology/scenario combinations. Fifteen
incorrect source variants were rejected. The receipt includes code and artifact
hashes, and distinguishes candidate failures from infrastructure failures.

One independent, workspace-only reader recovered the complete causal chain and
reported no direct answer pointer in the observations. This checks solvability,
not calibrated difficulty; it used instruction-level read isolation, not a
hardened container. The source still has one real whitening callsite, and an
experienced solver can find it from the launch configuration. No claim that
this revision meets a particular difficulty threshold follows from this check.

Run tests from the repository root with PyTorch and pytest installed:

```bash
JUSTRL2_SOURCE=/tmp/justrl2-source python -m pytest \
  proposals/miles_ppo_empty_shard_collective/justrl2/challenge/ -q
```

The source-integration tests require `JUSTRL2_SOURCE`; without it they explicitly
skip. They compare the actual advantage/return AST to the original paired
builder after removing observation calls: the defective and repaired
computations match their previously recorded counterparts.

For the native behavior checks, use the exported faulty workspace, then a copy
with `maintainer/repair.patch` applied. Keep verifier files and results outside
the source under test. The verifier runs CPU/Gloo collectives with functions
extracted from the candidate, varying group layout, inputs and masks. Its
numerical reference is independent of the candidate's output.

```bash
python /tmp/update-case/maintainer/topology_verifier.py \
  --source /tmp/update-case/workspace --output /tmp/observed-check
# Expected: exit 1, observed_intended_mismatch=true (the unmodified defect).

cp -R /tmp/update-case/workspace /tmp/update-repaired
git -C /tmp/update-repaired apply /tmp/update-case/maintainer/repair.patch
python /tmp/update-case/maintainer/topology_verifier.py \
  --source /tmp/update-repaired --output /tmp/repaired-check
# Expected: exit 0, passed=true, full_suite=true.
```

Default coverage is four layouts (CP1/2/4 × DP2, including permuted global
ranks) and nine scenarios: ordinary inputs, uneven valid counts, nonempty
zero masks, the original nondivisible token geometry, one valid token per DP
group, asymmetric CP emptiness, an empty DP replica, a wholly empty DP
coordinate, and globally zero masks. Global zero-count groups must retain the
native post-collective `ValueError`; returning zeros is not accepted. A separate
control group keeps peers alive to distinguish the intended Gloo timeout from
process teardown. Outer deadlines, missing ranks and source-load failures are
infrastructure errors, never reproduction evidence. Exit code 2 identifies
those errors; exit code 1 identifies a rejected candidate.

These checks do not import the entire distributed GPU stack or run a model.
The October 30 healthy/faulty/restore receipts are **not** new GPU verification
of this export. Before task acceptance, run the isolated image/asset preparation,
full GPU repair validation and blind solver/judge calibration, then integrate
and validate the formal task. No leaderboard or solve-rate result is claimed.
