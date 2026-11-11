# Original empty-shard case: multi-update symptom replay

This PR retains the original rank-local empty-shard guard. PR #30's separate
communicator-cache defect is not a dependency or part of this export.

The November 11 review noted that cleaning hints does not increase causal
difficulty. This revision changes the executed workload: the same four sample
copies run through multiple assignments and the native metric aggregator.
Two updates complete before the third fails. The first reported error can be
in downstream metric aggregation, although the initiating fault is an earlier
DP collective. These symptoms are produced by real CPU/Gloo execution.

## Build and run

Use Python 3.12+, CPU PyTorch/Gloo and NumPy and the pinned public JustRL-II
checkout at 75814318f243af7e39af337c1173fc3927c9635b:

    python export_challenge.py --source /path/to/JustRL-II --output /tmp/empty-case
    cd /tmp/empty-case/workspace
    python debug/run_updates.py --output /tmp/run-a

The exporter checks all 286 source hashes and the historical input pins.
Mount only workspace/ (or workspace.tar.gz) for the solver, without the parent
repository, maintainer/, source history or network access. The only native
defect is still the original guard; the reference patch removes it.

The solver gets the source, two authored trajectory inputs reused as four
copies, and run_updates.py plus its source loader. The runner records aggregate
status, completed-update count and the first actual exception's operation/type/
message. It omits Python caller frames, peer identities and later peer errors.
Worker stdout remains available for any solver-authored observations.
There is no trace switch, per-rank event ledger, intermediate tensor dump or
prescribed investigation method. Source print/logger calls and dormant dump
helpers remain removed, as in the previous export.

## Constructed workload and limits

This is a constructed assignment replay, not the pinned upstream sampler.
The global multiset (two copies of each original short/long sample), rewards,
masks and aggregate lengths remain identical at every update. Seeded contiguous
DP partitioning produces mixed batches for updates 0 and 1, then both short
samples on one replica at update 2. The total lengths are 424/357, while response
lengths are 2/329; they must not be conflated.

Under CP2 the short-only replica has an empty local response batch at CP1.
Its DP peer has nonempty responses. The original local guard skips whitening
on the empty rank. That rank and two others advance to a native CP×DP report
reduction; the nonempty peer is still waiting in DP whitening. A configured
one-second reporting deadline expires before the four-second compute deadline.
The default first exception therefore names the metric aggregator.

The timeout separation is deliberate and identical between defect and repair.
Controls change reporting/compute deadlines independently or make them equal;
ordering is not asserted for equal deadlines. Turning reporting off or extending
its deadline exposes the compute failure but does not repair it. An all-mixed
assignment and CP1 complete, even with the same global input lengths. The fixed
code completes all five original assignments. No sleeps or fabricated errors
are used to create this sequence.

The report values are synthetic squared-advantage/return summaries supplied to
the unchanged native aggregate_train_losses function. They use the native
sample-mean convention (full per-sample mask denominator, CP-local numerator,
replicated sample count). This is not the full training call order: no model
forward/backward or optimizer step occurs. The old GPU receipts do not validate
this new schedule or its report-stage timing.

## Independent reviewer checks

The detailed sequence_replay.py and sequence_audit.py stay outside the solver
mount. They retain the complete rank/update arrays and all timed errors.
A scalar implementation independently checks CP ownership, pre-normalized GAE,
whitening inputs/masks and DP group membership, normalized per-sample outputs,
returns and aggregate metrics. Missing ranks, corrupted outputs, wrong groups
and absent later peer errors fail the audit.

Reviewer controls cover the default defect and repair, mixed assignment, CP1,
disabled reporting, longer compute/report timeouts, equal timeouts and rank
permutation. The original numerical topology verifier remains available for
its broader mask/topology mutation coverage.

    JUSTRL2_SOURCE=/path/to/JustRL-II python -m pytest . -q

See staged_validation.json for fresh code/archive hashes and measured outcomes.
The older validation.json, instrumentation_validation.json and
instruction_validation.json describe their respective earlier exports.

Correct diagnosis and validation are graded on correctness. No log format,
instrumentation diff, experiment order or technique is required. A solver may
still infer the guard statically; this work does not claim impossible static
deduction or a calibrated solve rate. Blind-review findings are recorded
explicitly before claiming any difficulty improvement.


## Validation and blind review

The final revision passes **129 pytest cases plus 10 subtests**, with no failures
or skips. Nine staged control experiments pass the per-sample scalar audit.
The reference repair also passes the existing four-topology, 36-case verifier.
The final archive and scripts are bound by staged_validation.json.

One workspace-only blind solver explained the first reporting timeout and the
change from two completed updates to a failure. It reported that source reading
still directly exposed the guard; runs confirmed rather than enabled discovery.
Its synchronized-skip repair completed the visible sequence but was rejected by
the independent empty-DP-group case because it suppressed the native global
zero-mask error. These outcomes are not evidence that root location now meets a
target difficulty threshold; no calibrated solve-rate claim is made.
