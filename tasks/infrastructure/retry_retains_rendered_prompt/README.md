# Retry reset retains rendered prompt and blocks recovery

Training Qwen3.5-9B with GRPO on a mixed code-repair corpus. Retry and
recycle reset the generated state but preserve the rendered prompt string, so
every replay fails the messages-only input check and the rollout loop aborts
after 128 consecutive groups.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Independent |
| **Type** | Infrastructure |
| **Domain** | Code RL |
| **Tags** | `retry-lifecycle` `rendered-prompt` `abort-loop` |
| **Framework** | miles (GRPO), Megatron training, SGLang rollout |
| **Run time** | 10 hours 15 minutes |
| **Resources** | 16 × H100 (80 GB) |

## Difficulty

Generation accepts a messages list, renders it with the chat template, and
then overwrites `Sample.prompt` with the rendered string. The sample still
needs that rendered text for training, so the common retry reset deliberately
preserves `prompt`.

Both same-prompt retry and sandbox-failure recycle call `reset_for_retry()`.
That reset clears generated outputs but leaves the now-rendered string in
`prompt`. On replay, the messages-only input check raises
`native message input required`, the group is recycled again, and the replay
follows the same failing path. The recorded run completed 20 rollouts and 160
optimizer updates before the failure counter reached 128 and training stopped.

The optimizer metrics remained finite through the recorded window, the final
normal update reached step 159, and checkpoint markers existed for iterations
7 and 15. The later failure is therefore a lifecycle replay bug, not an OOM,
CUDA failure, network outage, or proof that the model stopped producing useful
trajectories.

A focused repair rebuilt the original messages from frozen task fields before
generation. It covered same-prompt retry, environment recycling, and direct
input of an already-rendered string; 56 tests passed. No clean same-seed,
same-budget post-fix training comparison is available.

Issue-only: the source repair has focused lifecycle tests but no uncontaminated
training control; a reward or step threshold would mix this fix with concurrent
network and environment changes.

## Reference solution

- **issue**: identify that `run_generation` replaces the messages list with
  rendered text, while `Sample.reset_for_retry` preserves `sample.prompt` and
  both retry paths call that reset; the replayed sample therefore repeatedly
  fails `validate_messages` before any service request. Reject the network,
  GPU, checkpoint, and “all tasks became unsolvable” diagnoses.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  a verdict score of 1.0 and root-cause and evidence scores above 0.8.
