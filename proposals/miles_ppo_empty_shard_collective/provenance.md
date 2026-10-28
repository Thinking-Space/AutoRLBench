# Evidence provenance

Prepared October 28, 2026 for a proposed AutoRLBench infrastructure task.

## Historical source

`historical_kernels.py` contains narrow excerpts from a historical Miles checkout
carrying an Apache-2.0 license. A copy is included as `LICENSE`. The original
checkout and deployment archive are private; the revision hashes are provenance
identifiers, not publicly fetchable asset URLs. This proposal does not publish
the rest of that checkout, model weights or private datasets.

`source_manifest.json` records original file SHA-256 hashes, exact excerpt line
ranges, excerpt hashes, the generated file hash and adapter changes. Function
bodies and the two normalization blocks are copied verbatim. The module wrapper
and `reproduce_gloo.py` are new standalone reviewer tooling. The original file
header/imports are not included in these excerpts; the included source license
applies. No upstream NOTICE file was present at the recorded source root.

Only the whitening block is compared between the two revisions. The first patch
also changed warmup checkpoint saving, which is outside this reproducer.

The adapters supply precomputed synthetic advantages, loss masks, sequence
lengths and a minimal parallel-state object. Gloo WORLD stands for one DP group
at a fixed CP coordinate. The original offset function and masking block are
retained; neither Megatron initialization nor a complete CP mesh is launched.

The adapter now also accepts an explicit DP process group for `reproduce_mesh.py`.
That optional parameter changes only the standalone wrapper, retaining WORLD as
the default for the earlier two-process reproducer. All six historical excerpts
remain verbatim; `source_manifest.json` records the current wrapper hash.

`reproduce_mesh.py` creates full CP×DP2 process groups (CP2 or CP4), checks each
group and CP reconstruction before whitening, and records per-rank inputs and
stages. It supports CPU Gloo or one CUDA device per process with NCCL. The
parent directly owns each worker process group and requires a target-rank
collective timeout, not merely a failed process or an outer deadline. This
driver is separate from the original two-process results below; GPU execution
results must be recorded separately.

## Historical logs

`historical_manifest.json` records SHA-256 and byte size of each complete original
master stdout file, selected original line numbers, the sanitization operations
and the published excerpt hash. The original files remain in the contributor's
private archive. Hashes anchor this extraction; they do not make the unpublished
full logs independently inspectable.

Selection includes every line in the two original files containing
`WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=3`. ANSI escapes are removed, private
addresses and process/thread identifiers replaced, and source runs renamed
`run_a`/`run_b`. Rank/group IDs, times, tensor counts, sequence numbers and timeout
measurements are unchanged. `[repeated ... across cluster]` is original Ray log
deduplication output, not an independently counted number of failures.

Task metadata was checked against the archived deployment records. `run_a` used
revision `5474ec1a087476347f024fea3c853ae3dae3df07`; `run_b` used
`35f2eea19fb210603c6ec4ba49bc978d867638ba`. Both allocated 96 H800 GPUs. Their
historical private image tag was `llm-rl/ngc-2506:nvidia.221a6cd3`; its digest and
an equivalent public image have not been established.

The archived deployed `loss.py` bytes were also read directly for both runs.
Their SHA-256 hashes match the corresponding pinned git source in
`source_manifest.json`: `1e4b771e...` for `run_a` and `f821ecd4...` for `run_b`.
This checks the source/installation boundary rather than relying only on a
mutable branch name. It does not recover the per-rank runtime inputs.

`runtime_arguments.json` contains only allowlisted startup arguments, with their
original stdout line numbers. Both logs confirm actor 6×8 / critic 6×8 GPUs,
TP1 / CP4, PPO, normalization enabled and global batch size 480. It is an extract
from the actual startup tables, not a reconstruction from script defaults.

The log date/clock fields are preserved as emitted; no timezone conversion is
applied to them. Historical task and source records date the failures to July
11–12, 2026. The code-level repair is not asserted to explain all those failures:
the post-patch recurrence is deliberately included.

## Newly executed reproducer

`reproducer_results.json` is actual output from eight separate two-process CPU
Gloo checks on October 28, 2026. Runtime: Python 3.12.3, PyTorch 2.11.0+cu129,
`CUDA_VISIBLE_DEVICES=""`, one torch/OMP thread per child. Each collective has a
three-second timeout and child processes have an outer cleanup deadline.

The inputs are isolated synthetic mechanism cases, explicitly separate from
historical training records. The old mixed case timed out; the repaired case
completed and matched pooled numerical reference values. Successful comparisons
use `rtol=1e-5`, `atol=1e-6`. The results include elapsed times for transparency,
not as a training-performance benchmark.

The original two-process CPU results do not reproduce the full 96-GPU execution,
the later GAE suspicion, model optimization, or held-out quality. No new RL
training was launched and no AutoRLBench score is claimed.

## Additional GPU mesh validation

`nccl_mesh_results.json` was collected from the completed October28 execution
of the final `reproduce_mesh.py`: one8-H100 node, real CP4×DP2 groups, torch2.7.1+cu126,
CUDA12.6,NCCL2.26.2. Its COMPLETE and EXPECTED_TIMEOUT records retain their original
SHA256 hashes, and the bundle includes32 healthy-control rank records plus the
eight pre-termination records for the defective mixed case. The original40rank
JSONs plus COMPLETE/EXPECTED_TIMEOUT were recovered from task stdout and all42
hashes checked before publication. Driver/source hashes match the tested files.

No model weights, training data or private addresses are included. Synthetic
input values, numerical outputs and watchdog timings are unchanged. Rank and
process-group information is retained. The single-node mesh shares the historical
CP size and reproduces the same three-element NCCL reduction signature, while
using DP2 and synthetic precomputed advantages instead of full96GPU training.

The first GPU attempt passed four controls and produced the expected timeouts,
but NCCL abort/trace cleanup exceeded the parent's deadline. It produced no
COMPLETE record and was not accepted. The final driver freezes pre-termination
evidence after every expected blocked rank reports the target watchdog, then
terminates/reaps its owned processes. Both checks around evidence collection
give the outer deadline priority. After a watchdog abort, no resumed CUDA output
is used as a numerical result.
