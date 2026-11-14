# Fully-async GRPO in-flight rollouts kept alive across weight sync

Training Qwen3-1.7B with fully-async GRPO on DAPO-Math-17k. The distributed
weight-update mixin silently rewrites the configured `abort` pause mode to
`in_place`, so in-flight rollout requests are kept alive across every weight
sync, decode to full length against stale weights, and are then dropped — the
engine burns decode on trajectories that never train, and the fixed 8-hour wall
clock completes 92 training steps instead of 148.

| | |
| --- | --- |
| **Author** | Qingfeng He |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `pause-mode-override` `inflight-rollouts` `throughput-regression` |
| **Framework** | miles (fully-async GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | 8 hours (fixed wall clock) |
| **Resources** | 4 GPU |

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation. The agent sees only `workspace/` and the one evaluation's
`instruction.md`; the fixed side, the oracle, and the measured runs all live
under `evaluations/`, which it cannot read.

Only the defective side is published. The fix is a single hunk in
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py`
(restoring the configured `abort`). (Provenance: the base is an internal
snapshot of the `miles` tree — the pinned commits are not reachable on the
public `radixark/miles` repo as of 2026-11-03, and public `main` has since
been restructured, so the control side was reproduced by patching the
shipped workspace rather than by cloning upstream.)

## Difficulty

The defect is six lines in
`DistBucketedWeightUpdateMixin._pause_and_prepare_engines`:

```python
mode = self.args.pause_generation_mode
if mode == "abort":
    mode = "in_place"          # <-- the bug: ignore the configured abort
ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
if mode != "in_place":
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
```

Nothing fails loudly. The launch config sets `--pause-generation-mode abort` and
the log faithfully echoes `pause_generation_mode ... abort`, so the config looks
correct — the code just doesn't honor it. Shapes are valid, training reward
climbs, and both arms learn at the same rate *per step*. The single strongest
tell is the recycling counters: `rollout/fully_async/stale_groups_recycled` sits
at ~32 (a full rollout batch) every step while `aborted_groups_recycled` stays
0. Interrupted groups are being recycled as **stale** (decoded to full length,
then dropped by the `--max-weight-staleness 0` filter) instead of **aborted**
mid-flight, so roughly half the engine's decode is spent on trajectories that
never enter training. A diagnosis that blames "the model collapsed" or "it
didn't learn" is wrong: both arms learn per step; the defect costs throughput.

| | defective | control |
|---|---|---|
| train steps in 8h | 92 | 148 |
| `stale_groups_recycled` / step | ~32 | ~0 |
| `aborted_groups_recycled` / step | 0 | ~32 |
| solve rate, last 10% of steps | 0.460 | 0.539 |
| solve rate, matched steps 0..91 (control − defective) | — | +0.05 (within ~0.08 per-step noise) |

## Reference solution

- **issue**: name the `abort -> in_place` override in `_pause_and_prepare_engines`
  (in-flight requests not aborted/evicted at weight sync, `flush_cache` skipped)
  as the root cause, connect it to the stale-recycling counters and the reduced
  step throughput, and reject "collapsed / didn't learn" and "config says abort
  so it's fine". Must not blame benign settings (`kl_loss_coef=0.00`,
  `max_weight_staleness=0`, the 4-GPU split).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict_score 1.0 with root_cause_score and evidence_score each above 0.8.
