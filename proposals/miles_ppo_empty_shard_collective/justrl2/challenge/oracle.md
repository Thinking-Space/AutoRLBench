# Maintainer-only diagnosis and repair contract

Expected verdict: unhealthy. The regression was deliberately inserted into
already-repaired public source. It is not an outstanding upstream defect.

## Necessary causal chain

1. Dispatch assigns sample 0 (424 total tokens, 2 response tokens) to DP
   replica 0 and sample 1 (357 total, 329 response) to replica 1. With zigzag
   CP2 response slicing, derive local response counts `[2, 0]` and `[149, 180]`
   from `get_logits_and_tokens_offset_with_cp` and the masks. Merely naming a
   short response without relating it to the slicing is incomplete.
2. In each role, CP groups are `[0,1]` and `[2,3]`; DP groups at fixed CP
   coordinate are `[0,2]` and `[1,3]`. The problematic pair is ranks 1 and 3,
   not ranks 0 and 1. Actor and critic have separate four-rank worlds.
3. In `compute_advantages_and_returns`, `all_masks.numel()>0` is a local
   predicate around `distributed_masked_whiten`. Rank 1 skips it and progresses;
   rank 3 calls the DP collective. Empty tensors and nonempty zero masks are
   different cases. GAE's preceding two CP reductions do not have this guard.
4. `distributed_masked_whiten` packs sum, squared sum and mask count into a
   fixed-size FP32 tensor and reduces over `parallel_state.intra_dp.group`.
   A tensor-empty member must still participate with zeros. Divergent
   participation causes the communicator/store failure; a later timeout is a
   symptom, and its communicator-local labels alone do not identify the ranks.
5. Remove the local conditional while preserving the same DP group, masks,
   unbiased pooled variance, GAE and returns. Changing to CP/WORLD, disabling
   normalization, filtering away the short replay, or increasing timeouts does
   not preserve the intended computation.

## Scoring contract for integration

Keep the current issue judge's verdict/root-cause/evidence weights 0.2/0.6/0.2
for the reported aggregate score. Passing is conjunctive: verdict must equal
1.0, root-cause must be strictly greater than 0.8, and evidence must be strictly
greater than 0.8, matching the existing issue `judge.py`. Root-cause credit is
at most 0.4 unless ALL causal links 1–4 are explained
correctly with concrete groups and local counts (equivalent derivations count).
Perfect verdict and evidence cannot compensate for incomplete root-cause credit.
A bare function name, grep result or correct one-line patch cannot pass the
diagnosis tier. Evidence credit requires distinguishing observations from
inference: the projected progress ledger does not itself prove entry into a
particular collective. This is an explicit rubric, not a calibrated LLM judge.

## Plausible alternatives

- A network/NCCL failure: consistent within-group divergence is predicted by
  dispatch, slicing and local control flow; raising the deadline cannot restore
  a missing participant. Retained evidence alone is not a network health audit.
- CP GAE reconstruction: the source places its two unconditional CP reductions
  before the guarded DP reduction; the groups differ. The observation view does
  not locate checkpoint 0 relative to these callsites, and a contestant must not
  be required to infer that hidden mapping or claim runtime proof of it.
- Bad rewards or zero loss masks: replay masks are all ones; the empty local
  tensor is created by slicing, not bad global data or mask values.
- A model/optimizer/attention failure: the first mismatched collective precedes
  the optimizer; changing unrelated components is unsupported.

## Repair verification

Apply `repair.patch` in a copy of the contestant workspace. Run the separate
`topology_verifier.py` against both versions, including topology permutations
and numerical negative controls. The verifier executes extracted native Python
functions with real Gloo collectives; it does not load model weights or verify
GPU forward/backward/checkpoint behavior. Existing October 30 GPU receipts apply
to the original instrumented pair. Do not relabel them as a new exported run.

The original full trace, rank events, pair generator and SHA inventories remain
in the proposal for reviewer audit. Never expose this file, the exporter,
`maintainer/`, original proposal, healthy checkout, source git history or an
installed repaired framework copy to the contestant. Disable network access.
Public publication inherently permits prior familiarity; measure it during
blind difficulty calibration instead of claiming contamination is impossible.
