# Sealed infrastructure oracle

## Verdict

**unhealthy.** A fully-async GRPO run on an otherwise-unmodified `miles`
checkout, with a single injected defect in the distributed weight-update mixin
that changes how in-flight rollout requests are handled at weight sync.

## Root cause

`inflight_not_aborted` —
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py::DistBucketedWeightUpdateMixin._pause_and_prepare_engines`
rewrites the configured `pause_generation_mode` from `abort` to `in_place`:

```python
mode = self.args.pause_generation_mode
if mode == "abort":
    mode = "in_place"
ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
if mode != "in_place":
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
```

With `in_place`, the rollout engines keep their in-flight requests alive across
the weight update instead of aborting/evicting them, and the `mode != "in_place"`
guard also skips `flush_cache`. Those interrupted trajectories decode to full
length against the old weights and are then dropped by the staleness filter
(`--max-weight-staleness 0`). The engine therefore spends full-length decode
compute on groups that never enter training.

## Required remediation

Remove the `abort -> in_place` override so `_pause_and_prepare_engines` honors
the configured `pause_generation_mode` (`abort`). Applying
`oracle/abort_mode.patch` to `workspace/` yields the control side. (Provenance
caveat: the base is an internal snapshot of the `miles` tree — the
`baseline_commit`/`buggy_commit` pinned in `oracle/task.yaml` were not
reachable on the public `radixark/miles` repo when last checked, 2026-11-03,
and public `main` has since been restructured, so the control side is
reproduced by patching the shipped workspace, not by cloning upstream.) Do not
disable
aborting by any other route (e.g. forcing `in_place` through a flag or raising
`--max-weight-staleness`).

## Observable impact

Evidence kind: `historical_run`. One measured run per side on 4×H800 for a fixed
8-hour wall clock (`TRAIN_SECONDS=28800`), same image and assets, same rollout
config; only the mixin override differs. Machine-readable copies under
`oracle/measured_runs/` and `oracle/historical_evidence_manifest.json`;
agent-visible frozen evidence for the defective side is in
`workspace/outputs/driver.log` and `workspace/outputs/rollout_data/*.pt`.

| metric | defective | control |
| --- | --- | --- |
| train steps completed in 8h | 92 | 148 |
| `stale_groups_recycled` / step | ~32 | ~0 |
| `aborted_groups_recycled` / step | 0 | ~32 |
| solve rate, last 10% of steps | 0.460 | 0.539 |

Per matched step the two arms are comparable (mean control−defective over the
overlapping steps 0..91 is +0.05, within per-step noise of ~0.08); the headline
solve-rate gap is downstream of the control completing more steps in the same
wall clock. "The run did not learn / collapsed" is a wrong attribution — both
arms learn per step; the defective arm completes fewer steps per wall clock.

## Machine grading

Machine grading reads only the new experiment logs under
`$AUTORLBENCH_RESULTS/experiment_logs`, via `tests/grade_submission.py` (spec in
`resources/grader_spec.json`). Two signals, **both** required:

- primary, train-step throughput: the number of completed training steps
  (largest `rollout N:` index + 1 in the driver log) must be at least
  **120** (defective 92, control 148).
- secondary, last-window solve rate: the mean fraction of samples with
  `reward.acc` over the last 10% of dumped rollout steps
  (`dump/rollout_data/*.pt`) must be at least **0.48** (defective 0.460,
  control 0.539).

Pass is score `1` (both thresholds cleared); otherwise `0`. The thresholds were
calibrated on one run per side, so the margins are reported without a variance
estimate; the primary train-step threshold is specific to the 4×H800 8-hour
setup and scales with grading hardware speed — see `verifier/README.md`.
