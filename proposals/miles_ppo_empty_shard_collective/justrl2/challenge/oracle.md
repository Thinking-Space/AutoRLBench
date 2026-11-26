# Maintainer-only diagnosis and repair contract

This is an authored regression inserted into already-repaired JustRL-II revision `75814318f243af7e39af337c1173fc3927c9635b`. PR #30 remains a separate communicator-cache defect.

## Root cause

`miles/utils/moments.py::MomentReducer` caches a zero FP32 vector of `[sum, sumsq, mask_count]` per device. It creates chunk triples lazily and computes `reduce(add, blocks, origin)`. For a nonempty input, `reduce` adds the chunks and returns a fresh tensor. For an empty input, it returns `identity` itself. The caller then passes that tensor to `dist.all_reduce`, which mutates the tensor in place. The empty worker's correct zero contribution is thereby replaced by its peer's aggregate statistics.

On a later nonempty input, the reducer uses the same mutated tensor as its starting value and adds the current chunks to stale prior statistics. The ensuing all-reduce combines old and current moments. Every DP member has entered the same collective, with the right group and tensor shape; the error is that one participant's local accumulator contains state from a prior update.

A valid repair must ensure the tensor passed to all-reduce is independently owned per invocation. The reference clones the cached identity for the empty case. Other equivalent implementations may also be valid if they preserve all moments and collective participation. Keep DP membership, masked sums, squared sums, count, Bessel correction, epsilon, returns and native zero-global-mask `ValueError` unchanged. Avoid persistent mutation of the reducer's identity.

## Default assignment and expected symptom

Two copies of each authored sample are shuffled with seed 3 and placed contiguously on the two DP replicas. The five batches are `[[3,0],[2,1]]`, `[[2,0],[3,1]]`, `[[0,1],[3,2]]`, `[[3,2],[1,0]]`, `[[3,1],[0,2]]`. The global sample multiset and all sample contents remain fixed.

Under CP2, short response shards have counts `[2,0]` and long response shards `[149,180]`. Updates 0 and 1 mix short/long responses on both replicas, so each CP rank has nonempty concatenated data. At update 2, one DP replica has only short samples; CP rank 1 has an empty local tensor. The same rank's reducer identity is mutated by the valid DP all-reduce. At update 3 that rank has nonempty data and its local triples include stale moments. The original policy-loss mismatch is first visible at update 3; later updates can remain affected. The first empty-shard update can itself agree with the independent oracle because the empty output contains no values.

The distinguishing history is: mixed calls, one locally empty but globally nonempty collective, then another nonempty call in the same worker process. A fresh standalone nonempty calculation, CP1, or an all-mixed schedule can pass. A prefix ending immediately after the empty update can also pass. These controls are intentional and should not be used alone to declare the repair correct.

## Evaluation

The private evaluator should compare per-update policy loss, value loss, clip fraction, local moment vectors and normalized advantages against the independent scalar oracle. It must test a continuous worker lifetime across the empty and subsequent nonempty calls. It must reject missing ranks, changed references, nonfinite outputs, wrong group membership and stale-state mutations.

The task does not require runtime discovery, logs, a fixed report schema, or specific debugging steps. Blind review must honestly report whether static reading found the cached identity alias and whether execution changed or confirmed the hypothesis. Difficulty calibration is empirical, not guaranteed by test count.

The historical one-update empty-shard collective defect is not the injected root in this export; retained GPU receipts are unrelated historical evidence. CPU/Gloo replay does not establish full model/optimizer/GPU training behavior.
