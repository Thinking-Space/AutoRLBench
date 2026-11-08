# Original empty-shard case: minimal initial evidence

This PR contains the original local-empty-shard collective defect only. The
separate communicator-cache case is submitted independently and is not required
to export or validate this task. Original GPU evidence remains in the parent
directory for reviewers.

The contestant starts with native defective source, replay inputs, launch
configuration and a minimal CPU runner. The export no longer contains the
precomputed CP/DP mesh, per-rank progress, backend exception, traceback, local
counts or numerical snapshots. Those artifacts remain outside the contestant
mount. All standalone print/logger calls and non-license comments in copied
framework Python files are removed uniformly; computational identifiers,
branches, asserts, docstrings and arithmetic are retained.

There is no prebuilt tracing option or hidden switch exposing diagnostics.
`debug/run_replay.py` reports aggregate completion and captures process stdout
for observations the solver writes. It does not add GAE/whitening hooks or save
intermediate tensors, group membership or rank-state reports. Internal coarse
worker statuses are temporary and are deleted after aggregation.

## Export and reproduce

Use the pinned public source, Python 3.12+, CPU PyTorch and NumPy:

```bash
python export_challenge.py --source /path/to/pinned/JustRL-II --output /tmp/empty-case
cd /tmp/empty-case/workspace
python debug/run_replay.py --replay inputs/replay.json --cp 2 --dp 2 --output /tmp/run-a
```

The source revision remains `75814318f243af7e39af337c1173fc3927c9635b`.
All 286 source hashes and original evidence pins are checked before export.
Only `workspace/` or `workspace.tar.gz` may be mounted for a solver; omit this
repository, parent folders, source git history and `maintainer/`, and disable
network access. The original source pin already repairs the defect, which the
case deliberately reintroduces.

The instruction states the failure, execution requirements and deliverables:
a repair and a supported explanation. It does not prescribe investigation
steps, tools, logging, assertions, an instrumentation diff or an evidence schema.
Reviewers assess causal accuracy, repair correctness and support for claims;
they must not require a particular diagnostic method.

## Reviewer checks

`maintainer/repair.patch` removes the unchanged local guard. The separate
`topology_verifier.py` executes native functions over real Gloo groups and checks
CP slices, PPO GAE, pooled unbiased variance and returns. It tests the repair
across layouts, masks and empty inputs, and rejects the original defect and
plausible wrong fixes. Its hooks, expected answers and detailed events are not
exported to the solver. Full historical traceback and records are not deleted.

```bash
JUSTRL2_SOURCE=/path/to/pinned/JustRL-II python -m pytest . -q
```

Tests compare the exported computation to the originally recorded pair after
removing observations, scan the contestant archive for supplied diagnostics,
check deterministic packaging and run real failure/repair cases. An integration
test adds a temporary observation to a copy of the candidate and verifies that
its output becomes available without repairing the defect; it is never shipped
as a solver helper.

The previous [validation receipt](validation.json) describes the earlier,
more informative November 7 export. New results are recorded separately in
`instrumentation_validation.json`. Neither receipt claims a GPU rerun of the
new presentation. The historical 8-H100 evidence validates the original pair,
not all source-cleanup transformations. CPU function checks do not cover full
framework imports or model/optimizer execution.

Removing supplied diagnostics increases the investigation required to obtain
runtime evidence, but an expert may still infer the local-guard bug statically.
No claim of impossible static deduction or calibrated benchmark difficulty is
made. Investigation choices belong to the solver; validation is assessed by
its relevance and correctness, not by the diagnostic method used.

The export also disables diagnostic-only backward hooks and output helpers,
removes bundled debug scripts, and retains inert import-compatible dumper APIs.
Environment-triggered gradient/tensor dumps cannot be enabled from the supplied
source. Core PPO numerical functions are unchanged apart from observations;
full GPU framework import behavior after this cleanup is not revalidated.

Current validation: **112 pytest cases and 10 subtests passed**, with no
failures or skips. See `instrumentation_validation.json` for source/archive hashes.

The subsequent method-neutral instruction revision passed 16 export tests.
Only `instruction.md` changed in the contestant workspace; runtime code and
initial evidence are byte-for-byte unchanged. See `instruction_validation.json`.
