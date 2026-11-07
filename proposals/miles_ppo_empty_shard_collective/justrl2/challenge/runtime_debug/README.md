# Runtime debugging variant

This candidate replaces the conspicuous local-empty guard with a separately
constructed, stateful communication-plan regression. It is an **alternative
case**, not a harder presentation of the same historical GPU failure. The
parent directory keeps the historical case and its original evidence intact.

A static-only re-review of the November 7 workspace recovered the guard and
the full rank/count explanation without executing code. That version therefore
does not meet the request for a debugging exercise. This variant removes the
guard entirely and supplies an actual, bounded CPU/Gloo debugging environment.

## What changes

Native CP slicing, PPO GAE and DP whitening remain unconditionally invoked.
Their SUM reductions pass through a small reusable plan that binds a native
communicator. Its defective cache key describes backend, size, local rank,
device, dtype and element count, but omits communicator identity. Different
groups can collide even though each call is locally well formed.

The probe fixes the batch size and varies response lengths and group layout.
GAE reduces `[B,max(response_lengths)]` while normalization reduces three
statistics. Payload size, CP/DP coordinates, and which plan was cached first
jointly determine the first misrouted operation. Thus the final timeout alone
does not locate the cause. Cold and warm experiments separate the hypotheses.

No fake log events, dummy functions, symbol scrambling or artificial delays
are introduced. The faulty and reference-correct sources differ only in the
plan key. Correct group-aware keys, safe late binding, or removing the cache
are acceptable if routing and numerical behavior remain correct.

## Measured verification

The [November 8 receipt](validation.json) records **155 passing tests, zero
failures and zero skips**: 104 existing tests and 51 new runtime tests. These
results include independent numerical checks, actual collective coverage,
identity/shape/membership cache mutations, imported-alias repairs, source-change
rejection and bounded CLI inputs.

Fresh runs against the exported native source gave these outcomes:

| Experiment | Defective plan | Reference repair |
| --- | --- | --- |
| CP2×DP2, long input only | Completes with correct numerics | Pass |
| CP2×DP2, cold short input | First wrong route at whitening; actual Gloo timeout | Pass |
| CP2×DP2, long then short | First wrong route moves to GAE; actual Gloo timeout | Pass |
| CP2×DP2, short then long | Wrong route on the first update | Pass |
| CP1×DP2, long then short | Completes with correct numerics | Pass |
| CP2×DP3, long then short | Completes with correct numerics | Pass |
| CP2×DP2, permuted ranks | Wrong route follows the new membership | Pass |
| CP4×DP2, long inputs | Completes with correct numerics | Pass |
| CP4×DP2, new lengths/seed, sparse masks, permuted ranks | Completes with correct numerics | Pass |

All nine repaired experiments also pass the independent scalar checks. The
native-communicator identity gate rejects the defective plan and accepts the
repair. The symptom is deterministic for a given workload/layout/history, not
an artificially randomized race. Short inputs with some other layouts can
legitimately produce a globally empty DP mask; those validation errors are not
used as evidence of this routing defect.

## Export and debug

With Python 3.12+, CPU PyTorch/Gloo and NumPy, from this directory:

```bash
python export_runtime.py --source /path/to/pinned/JustRL-II --output /tmp/runtime-case
cd /tmp/runtime-case/workspace
python debug/replay_probe.py --output /tmp/probe-a --cp 2 --dp 2 --steps 11,3
python debug/replay_probe.py --output /tmp/probe-b --cp 2 --dp 2 --steps 11,3 --trace
```

The public source pin and complete file inventory are inherited from the parent
exporter. This export contains no historical observations, GPU launcher, local
empty guard, reference patch, verifier or upstream git history. It emits a
separate `maintainer/` with hashes, oracle and reference patch. Mount only
`workspace/` for the solver, with no network or parent/source checkout mounts.

Each command starts fresh processes; steps within one command share state.
Options vary lengths, CP/DP layout, rank ordering, seed and masks. Argument
bounds prevent accidentally launching unbounded replay workloads. Tracing is
off by default; enabling it records actual requested/effective group members,
per-rank sequence, shape, phase and enter/return/error events. The probe has no
expected-answer logic. All counts, outputs and errors are produced by execution.

The instruction requires fresh failure and passing controls, the first
misrouting and its earlier priming call, a correction, and repaired runs with
numerical preservation. A static guess or patch without experiments is an
incomplete debugging submission. `grade_debugging.py` checks the declared
`experiments.json` artifacts: two history-dependent failures with different
first failure phases, at least two passing controls, every experiment repeated
after repair, and a new post-repair configuration. It rejects missing traces,
stale/wrong source hashes, absent controls and patches without run artifacts.
This is an executable evidence-consistency gate, not tamper-proof attestation.
Combine it with fresh trusted replay and semantic report review when integrating
the task into AutoRLBench; no formal task registration is claimed here.

## Maintainer acceptance

Use the trusted runner from this repository, not an edited solver copy:

```bash
python verify_runtime.py --source /tmp/runtime-case/workspace --output /tmp/before
# Defective source is rejected; passing controls and actual failure evidence remain.
cp -R /tmp/runtime-case/workspace /tmp/runtime-corrected
git -C /tmp/runtime-corrected apply /tmp/runtime-case/maintainer/repair.patch
python verify_runtime.py --source /tmp/runtime-corrected --output /tmp/after
python grade_debugging.py --manifest /path/to/submission/experiments.json \
  --original /tmp/runtime-case/workspace --repaired /tmp/runtime-corrected
```

The verifier uses independent scalar GAE, CP response ownership and unbiased
pooled variance; it checks raw and normalized advantages, returns and source
hashes. It rejects stopping at liveness, changing normalization groups, disabling
normalization, generic timeout evidence and missing ranks. A separate real Gloo
test uses distinct communicators with identical memberships and tensor shapes
to reject shape-only or membership-only cache keys. That closes shortcuts which
can pass just the visible PPO replay.

Run tests with `JUSTRL2_SOURCE` pointing at the pinned public checkout:

```bash
JUSTRL2_SOURCE=/path/to/pinned/JustRL-II python -m pytest . -q
```

## Scope

The inputs use synthetic critic values and authored rewards. Native functions
are loaded through AST selection; the complete new plan module is executed so
its globals and cache lifetime are real. This does not test model forward or
backward, optimizer updates, GPU/NCCL behavior or full framework imports.
The old October 30 GPU results do not validate this new defect.

Static source inspection can still suggest the faulty cache key. No finite
source-code task can guarantee that expert static deduction is impossible.
The added substance is experimentally distinguishing shapes, topology and call
history, and preserving numerical semantics. Only measured blind runs can
establish whether that is sufficiently difficult for the benchmark; no target
solve rate or final difficulty tier is claimed.

One blind solver inspected only the exported workspace, ran 13 bounded CPU
experiments and repaired the plan. Its seven repaired replays passed; the
maintainer independently checked its three failure traces, three controls and
seven post-repair artifacts through the evidence gate. The solver explicitly
reported that static reading already identified the cache-key issue and
predicted the phase change; experiments confirmed and quantified it. Therefore
this exercise **does not prove that runtime debugging is necessary to discover
the root cause**, nor that the benchmark's difficulty threshold is met. Runtime
evidence is required for the submission protocol, which is a narrower claim.
