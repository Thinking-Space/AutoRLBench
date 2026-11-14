# Maintainer-only runtime regression oracle

This is a new, deliberately constructed cache-routing regression, not the
historical local-empty guard and not a new failure of untouched upstream code.
The old 8-H100/GPU receipts do not apply. All evidence must come from new runs.

The reusable SUM plan in `miles/utils/collective_plan.py` binds a concrete
process group into `partial(dist.all_reduce, group=...)`. Its lookup signature
includes backend, group size, communicator-local rank, device, dtype and element
count, but omits communicator identity. Two distinct groups can have identical
signatures. CP and DP are different memberships even when both have size two.

Native GAE allocates `[B,max(response_lengths)]`. This probe uses `B=1` and
float32 for both values and rewards. It first reduces a `[1,L]` values buffer
and a rewards buffer over CP.
Native whitening reduces `[sum, sumsq, count]` over DP. A three-token response
makes both payloads have three elements. In CP2×DP2, workers with equal CP/DP
local coordinates reuse a plan bound to the wrong group; others use the requested
group. Participation diverges even though every native call is unconditional.
The first operation that misroutes depends on prior calls: cold short input
primes a CP plan before whitening; a long update primes DP's three-element plan,
so a subsequent short input can misroute at GAE. A short sample alone is not
enough: cache state must persist and dtype, device, backend, group size and local
rank must also match. Diagnose this with requested
versus effective members, not the ultimate timeout traceback. Capture actual
ordered traces for each scenario; do not assume the same waiting rank/callsite.

Correctly include the process-group handle in plan identity, or remove caching
while preserving each requested group and numerical operation. Cached generic
operations with the requested group passed at invocation (safe late binding)
are also valid. Judge behavior, not similarity to the reference patch. Group size,
local rank, tensor shape, or membership alone does not identify a distinct
communicator. Changing CP/DP groups, disabling normalization or filtering inputs
is not a repair. Clearing a cache at one convenient boundary is not a general
identity fix. Native empty local tensors remain legitimate participants.

For acceptance require a correct explanation of the communicator identity
collision and its conditions, a repair preserving each requested group, and
supporting validation. A bare cache-key guess is not a complete explanation.
Do not prescribe how the solver investigates, a number or order of experiments,
logging, trace fields or a particular artifact schema. Reviewer-side tracing
can independently check the repair; it is not a required solver deliverable.
The optional AST replay does not validate full GPU training.

Static inspection can suggest this bug. No finite source task can honestly
guarantee that experts cannot deduce it; the report and independent
verification must be judged on correctness, not on investigative technique. Report blind
review outcomes rather than claiming an unmeasured difficulty threshold.

## Method-neutral evaluation

The initial export supplies only minimal replay status. It does not tell the
solver what information is absent or suggest how to acquire more information.
Accept any sound investigative approach. Do not require an instrumentation-only
diff, custom logs, trace files, experiments.json or use of grade_debugging.py.
That tool is an optional reviewer audit of reviewer-generated experiments;
verify_runtime.py and plan_identity.py provide independent correctness checks.
