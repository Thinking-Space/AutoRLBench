# JustRL2 communicator-plan cache case

This PR contains only the constructed communicator-cache regression. It is
independent of the historical empty-shard proposal: exporting, testing and
validating it does not import files or require a commit from that PR. Its
source pin, source inventory, license and native test fixtures are self-contained.

Native PPO/GAE/normalization calls execute unconditionally. The faulty plan cache
binds a process group but looks up plans using a descriptor that omits its
identity. Matching payload features and CP/DP local coordinates can select a
plan bound to a different communicator. Call history determines whether the
first wrong route occurs in GAE or normalization. This is a deliberately
constructed bug, not a claim about an untouched upstream revision.

## Contestant boundary

Only the native defective source, input generation and `debug/run_replay.py`
are exported. Copied framework Python has no supplied logger/print calls or
non-license comments. The new cache module contains no diagnostic logging.
The runner has no tracing option, requested/effective-route events, phase hooks,
intermediate tensors, per-rank status files or automatic error traceback.
It writes aggregate completion and captures stdout/stderr for observations the
solver adds. Internal coarse statuses are deleted after aggregation.

The instruction states the failure, execution requirements and deliverables:
a repair and a supported explanation. It does not prescribe investigation
steps, tools, logging, assertions, an instrumentation diff or an evidence schema.
Full tracing and numerical reference code below is reviewer-only, not a hidden
flag in the solver's runner. Reviewers must not require a particular diagnostic
method.

## Build and run

Python 3.12+, CPU PyTorch/Gloo and NumPy are required. Use public JustRL-II commit
`75814318f243af7e39af337c1173fc3927c9635b`; all 286 source hashes are checked.

```bash
python export_runtime.py --source /path/to/pinned/JustRL-II --output /tmp/plan-case
cd /tmp/plan-case/workspace
python debug/run_replay.py --cp 2 --dp 2 --steps 11,3 --output /tmp/run-a
```

Only `workspace/` or its deterministic archive belongs in the agent mount.
Do not expose this repository, maintainer directory, git history or a repaired
source checkout, and disable network access. The raw public source remains
outside the contestant container. No model weights or prior GPU evidence ship.

## Reviewer validation

The trusted `replay_probe.py` has detailed instrumentation for the reviewer.
`verify_runtime.py` uses it with independent scalar GAE, CP ownership, unbiased
pooled variance and return checks across nine scenarios. `plan_identity.py`
checks distinct native communicators with identical membership and tensor shape.
These tests reject shape-only/member-only cache repairs while accepting correct
group identity, uncached execution and safe late binding. Original and repaired
sources differ only in the group key.

```bash
python verify_runtime.py --source /tmp/plan-case/workspace --output /tmp/before
cp -R /tmp/plan-case/workspace /tmp/plan-repaired
git -C /tmp/plan-repaired apply /tmp/plan-case/maintainer/repair.patch
python verify_runtime.py --source /tmp/plan-repaired --output /tmp/after
JUSTRL2_SOURCE=/path/to/pinned/JustRL-II python -m pytest . -q
```

`grade_debugging.py` checks consistency of reviewer-produced full-trace
experiments. It does not prescribe a solver log schema or prove that files
cannot be fabricated. This optional reviewer audit is not a solver submission
gate; its prescribed experiments and artifact format are not requirements for
acceptance. Evaluate the report and rerun the repair with trusted tools. The previous
[validation](previous_validation.json) concerns the earlier instrumented export;
new sparse-observation results are in `instrumentation_validation.json`.

The new export tests prove that pristine runs provide only coarse status,
that source-added logging yields runtime evidence without repairing the bug,
and that the corrected case still passes independent numerical checks. All
changes to copied source are recorded by before/after hashes in the manifest.

## Limits

This is a CPU/Gloo native-function replay with synthetic values and authored
rewards, not GPU/NCCL training, model execution or an optimizer/checkpoint test.
The complete cache module executes, but native functions use AST selection;
whole-framework import behavior is outside this check.

An earlier blind solver found the key error statically and used existing tracing
to verify it. This revision removes that supplied tracing while leaving
investigation methods unrestricted. It does not claim that expert static deduction
is impossible, or that a particular difficulty/solve-rate target is calibrated.
No formal AutoRLBench task or integrated judge is claimed yet.

The export also disables diagnostic-only backward hooks and output helpers,
removes bundled debug scripts, and retains inert import-compatible dumper APIs.
Environment-triggered gradient/tensor dumps cannot be enabled from the supplied
source. Core PPO numerical functions are unchanged apart from observations;
full GPU framework import behavior after this cleanup is not revalidated.

Current validation: **59 pytest cases and 10 subtests passed**, with no
failures or skips. See `instrumentation_validation.json` for source/archive hashes.

The subsequent method-neutral instruction revision passed 3 export tests.
Only `instruction.md` changed in the contestant workspace; runtime code and
initial evidence are byte-for-byte unchanged. See `instruction_validation.json`.
