# Proposal: Miles PPO empty-shard whitening collective mismatch

An RL training process can time out in advantage normalization when context
parallelism leaves one data-parallel worker with no local response tokens and
that worker skips a collective required by its peers.

**Status:** task proposal with historical evidence, CPU/GPU reproductions, and
real healthy/faulty training plus native checkpoint restoration on a public
image; **not a packaged or leaderboard-ready AutoRLBench task**.

The [public JustRL2 case](miles_ppo_empty_shard_collective/justrl2/README.md)
builds paired workspaces from pinned open-source code and public model assets.
The pinned source already fixes the defect; the faulty arm explicitly
reintroduces the empty-shard guard. Its fixture contains authored math
trajectories, without fabricated rollout log probabilities.

On October 30, 2026, the final public image completed the healthy actor4/critic4
training path on eight H100s with CP2/DP2, while the faulty arm produced the
expected first divergence and target collective failure. A separate restore
stage loaded exact native model/Adam state on all eight ranks, advanced step 1
to step 2, saved checkpoint 1 and verified its payload against live state.
Original checkpoint hashes remained unchanged. A separate CPU task reran the
acceptance checks and rehashed both checkpoint generations. These final-image
results are distinguished from the earlier internal-runtime runs and the
historical CP4 kernel reproducer below. See the
[final receipt](miles_ppo_empty_shard_collective/justrl2/public_runtime_results.json).

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

### Additional NCCL validation (October 28, 2026)

The [full mesh driver](miles_ppo_empty_shard_collective/reproduce_mesh.py) was run
on one node with **8 NVIDIA H100 80GB GPUs, CP4 × DP2**, using PyTorch 2.7.1+cu126,
CUDA 12.6 and NCCL 2.26.2. The [recorded results](miles_ppo_empty_shard_collective/nccl_mesh_results.json)
include all per-rank inputs, topology, reference outputs and timeout evidence.

- Original/nonempty, repaired/nonempty, repaired/zero-mask-nonempty and
  repaired/mixed-empty each completed with correct values on all eight ranks.
- Original/mixed-empty produced the actual NCCL watchdog signature
  `SeqNum=2, OpType=ALLREDUCE, NumelIn=3, NumelOut=3` on ranks 5, 6 and 7 after
  15,046 / 15,068 / 15,080 ms, with a configured 15-second timeout.
- CP0's DP group completed numerically; ranks 1, 2 and 3 had empty local slices
  and skipped whitening. Each blocked peer was in the expected corresponding DP
  group. Group communication and CP reconstruction passed before the target call.
- After all three watchdog records were captured, the parent froze the evidence
  and terminated/reaped its own workers. Their `-15` exit codes mean intentional
  cleanup after the demonstrated failure, not normal completion of the bad code.

The driver writes `COMPLETE.json` only when all five expected outcomes hold. This
run completed at **2026-10-28 09:35:33 UTC**. It is one execution of the final
driver, not a multi-seed training experiment. An earlier GPU attempt obtained the
same three timeouts and four successful controls but exceeded the outer cleanup
deadline; it was rejected and is not counted as a passing run. The final driver
accepts evidence only before that deadline and does not resume CUDA computation
after an observed watchdog failure.

This adds actual GPU/NCCL evidence for the isolated code defect. It does not
establish the complete cause of every historical 96-GPU failure or a repaired
end-to-end RL training result.

From the repository root, on a node with eight available GPUs:

```bash
python proposals/miles_ppo_empty_shard_collective/reproduce_mesh.py \
  --backend nccl --cp-size 4 --output /tmp/cp-whitening-nccl-run
```

Use a new output directory per execution. The driver also accepts `--backend gloo`
for a CPU mesh preflight.

### Intended benchmark contract

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

The public image, pinned model/data preparation, paired real training and native
checkpoint restoration have passed. Registration still requires:

- Agree with maintainers on the task scope and resource budget.
- Separate the defective-only agent workspace from the hidden repair/verifier.
- Add hidden numerical/adversarial cases and calibrate bounded completion and
  numerical-correctness acceptance.
- Resolve the pinned model's use terms before final acceptance; no weights or
  checkpoint tensors are redistributed in this proposal.
- Run AutoRLBench's validator and environment preparation after creating a
  formal task directory. Neither is claimed for this proposal.

See [provenance](miles_ppo_empty_shard_collective/provenance.md) for source hashes,
extraction and sanitization details. This PR adds no entry under `tasks/` and
does not change the leaderboard.
