# Proposal: Miles PPO empty-shard whitening collective mismatch

An RL training process can time out in advantage normalization when context
parallelism leaves one data-parallel worker with no local response tokens and
that worker skips a collective required by its peers.

**Status:** task proposal with recovered historical evidence and an executable
CPU reproducer; **not a packaged or leaderboard-ready AutoRLBench task**.

| Field | Value |
| --- | --- |
| Proposed domain | Infrastructure |
| Proposed task ID | `miles_ppo_empty_shard_collective` |
| Historical framework / algorithm | Miles + Megatron, critic-based PPO |
| Historical workload | MiniCPM5-2.6B, long-context mathematical reasoning |
| Historical deployment | 96 H800 GPUs, actor 48 / critic 48, TP1 / CP4 |
| Reproducer | Two CPU/Gloo processes representing one DP group at CP coordinate 1 of CP2 |
| Initial evaluation target | Issue; Solution after agreeing on a liveness + numerical-correctness contract |
| Contributor | Pan Haoxuan (`merak0514`) |

## Why this would be a useful task

The visible error is a generic NCCL watchdog timeout. The relevant defect is a
local-data-dependent branch around a distributed statistics reduction, not a
long-running GPU kernel or an insufficient watchdog timeout.

An agent must follow the chain from variable-length responses through CP slicing,
identify which ranks share the **DP** statistics group, and realize that a rank
with no local tokens must still contribute zero statistics. Identifying that
CP is involved without locating the DP participation mismatch is insufficient.

The case also tests evidence discipline: the same historical timeout recurred
after the first patch, so one recognizable error signature cannot establish one
complete explanation for every failed run.

## Source-confirmed defect and repair

In historical `miles/backends/training_utils/loss.py`,
`compute_advantages_and_returns` gates whitening with:

```python
if all_masks.numel() > 0:
    whitened_advs_flat = distributed_masked_whiten(
        all_advs, all_masks, process_group=parallel_state.intra_dp.group,
        shift_mean=True,
    )
```

`distributed_masked_whiten` in `miles/utils/distributed_utils.py` reduces a
three-element tensor containing the masked sum, masked squared sum and mask sum.
CP slicing can produce an empty local tensor. Different DP replicas can receive
different response lengths, so the branch can differ among members of the same
DP group.

The scoped repair removes the local participation guard. An empty local tensor
contributes `[0, 0, 0]`; all group members execute the reduction. A globally empty
mask remains an explicit error. Disabling normalization or substituting local
statistics would change the training computation and is not an equivalent fix.

The accompanying [source excerpts](miles_ppo_empty_shard_collective/historical_kernels.py)
contain the original and repaired normalization blocks, the statistics function,
and the CP offset helpers. These are **reviewer-facing evidence**, including the
answer; they must not be mounted wholesale into an eventual agent workspace.

## Historical evidence and its limits

Two original master stdout files were recovered from the training archive.
Their [startup arguments](miles_ppo_empty_shard_collective/runtime_arguments.json)
confirm the parallel configuration and enabled advantage normalization.
The [sanitized excerpt](miles_ppo_empty_shard_collective/historical_excerpt.log)
preserves all lines in those files matching this timeout signature:

```text
WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=3, NumelOut=3, Timeout(ms)=600000)
```

| Evidence | Revision / date | What it establishes |
| --- | --- | --- |
| `run_a` | `5474ec1a087476347f024fea3c853ae3dae3df07`, July 11, 2026 | A real 96-GPU PPO run with the guarded implementation encountered this timeout. |
| First patch | `35f2eea19fb210603c6ec4ba49bc978d867638ba`, July 12 | Removes the whitening guard. The full commit also changes warmup checkpoint handling; that unrelated change is excluded from the reproducer. |
| `run_b` | `35f2eea19fb210603c6ec4ba49bc978d867638ba`, July 12 | The same timeout signature occurred after the first patch. This is a failed post-patch run, not a healthy control. |
| Diagnostic follow-up | `c1d3d8e4c61a400736df9728c06e4a5f986aa81c`, July 12 | Its commit message acknowledges recurrence and adds per-rank GAE/whitening diagnostics. |
| GAE follow-up | `571657797ebf067344983374070326838e6d1264`, July 12 | Replaces per-sample CP gathers with two batched reductions; its commit message identifies variable communication count as a suspect, not a proven full incident explanation. |

The source and reproducer establish a specific participation defect. The
historical excerpt establishes that the training failures occurred, but does
**not** identify the absent ranks or prove that this defect caused every archived
timeout. There is no claim here of a verified healthy full-training control.

## Reproduce the scoped defect

With an existing PyTorch installation supporting CPU Gloo, from the repository
root:

```bash
CUDA_VISIBLE_DEVICES="" python proposals/miles_ppo_empty_shard_collective/reproduce_gloo.py
```

Tested on Linux, Python 3.12.3, PyTorch 2.11.0+cu129, using CPU only. This recorded
environment is separate from the historical NCCL training environment. No model,
dataset, GPU, network service or training allocation is needed; Gloo uses local
process communication.

The two processes represent the **same CP coordinate in different DP replicas**,
not two members of a CP group. Both use total length 16. At CP2/rank1, response
length 2 gives zero local response tokens while response length 8 gives five.
The original CP offset and masking code derives these slices. Advantages are
explicit synthetic inputs to isolate normalization; no GAE or model forward pass
is executed.

The driver requires the original mixed case to produce a real collective timeout,
then checks repaired outputs against independently pooled mean and unbiased
variance. It keeps the skipped peer alive until the timeout and bounds child
process lifetimes. The driver exits successfully only when all expected
differential outcomes match; success does not mean the original implementation
is healthy.

Recorded [results](miles_ppo_empty_shard_collective/reproducer_results.json), one
execution of each of these eight variant/case combinations:

| Case | Original | Repaired |
| --- | --- | --- |
| One empty and one nonempty local shard | Nonempty peer times out after approximately 3 seconds | Both finish; normalized values match the reference |
| Both local shards nonempty | Numerical reference matches | Numerical reference matches |
| Nonempty tensors, one rank's mask all zero | Numerical reference matches | Numerical reference matches |
| All local shards empty | Silently skips global validation | Both reject the globally empty mask with `ValueError` |

The all-zero-but-nonempty case matters: the faulty branch checks `numel()`, not
`mask.sum()`. The reproducer should not imply that every zero-valued local mask
causes this hang. These are deterministic mechanism checks, not multi-seed RL
training results or a GPU performance measurement.

## Proposed evaluation

For **Issue**, ask the agent to diagnose a stalled PPO run using a frozen source
snapshot, effective parallel configuration, relevant logs and representative
batch-shape evidence. The hidden oracle should require the local branch, the DP
statistics collective and the CP/response-length mechanism. It should distinguish
the demonstrated code defect from unsupported claims about all historical hangs.

For **Solution**, a possible acceptance contract is bounded completion with
correct pooled whitening, including empty-local/nonempty-global and ordinary
nonempty cases. Globally invalid inputs must remain explicit errors. The harness
should rerun the candidate code with private cases, rather than trust an agent's
reported success or accept a disabled normalization path.

This is a proposed contract for maintainer discussion, not an implemented grader.
The present reviewer driver deliberately contains both reference variants and
cannot be reused as an agent-visible benchmark harness without separation.

## Remaining work before task registration

- Confirm whether a source-derived distributed kernel repair is in scope, or
  whether the task must include a complete RL training replay.
- Select or build a shared benchmark image and pin dependencies. The private
  historical image tag alone is not a publicly reproducible environment.
- For full RL replay, provide accessible pinned model/data assets and restore the
  exact launch/runtime configuration and appropriate batch evidence.
- Construct a defective-only workspace and hidden oracle/verifier; calibrate a
  bounded completion and numerical-correctness evaluation with maintainers.
- Obtain a clean repaired full-training control before making any end-to-end
  recovery or training-quality claim.
- Run the AutoRLBench validator and environment preparation once a real task
  directory exists. Neither is claimed for this proposal.

See [provenance](miles_ppo_empty_shard_collective/provenance.md) for source hashes,
extraction and sanitization details. This PR adds no entry under `tasks/` and
does not change the leaderboard.
