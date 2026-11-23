# Threaded reward calls return scores in finish order

Qwen2.5-1.5B-Instruct is trained with TRL GRPO on GSM8K, and the answer
checker for each completion runs in its own child process. The workspace's
vendored TRL has a local `reward_num_workers` option that calls the reward
function on slices of the batch from a thread pool. It concatenates the
slices with `concurrent.futures.as_completed`, so the scores come back in the
order the threads finish and the trainer attaches them to the wrong
completions. The batch reward drifts from 0.07 down to 0.001 and greedy GSM8K
test accuracy goes from 0.017 to 0.000. The same workspace with the scores
collected in submission order reaches 0.75.

| | |
| --- | --- |
| **Author** | Yaolun Zhang |
| **Affiliation** | Oregon State University |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | `reward-misalignment` `concurrency` `completion-order` `vendored-framework` `silent-failure` |
| **Framework** | TRL 0.27.2, vendored in `workspace/trl` with a local patch (GRPO, `dr_grpo` loss, colocated vLLM 0.11.0); transformers 4.57.6, peft 0.19.1 |
| **Run time** | ~16 minutes on an H100 NVL (two 1,319-item evaluations plus 200 training steps) |
| **Resources** | 1 × H100 NVL (94 GB), 12 CPU cores |

## Difficulty

The `as_completed` reordering in `_call_reward_func_concurrently` is a
deliberately injected local modification to the vendored TRL tree, not a claim
of an unfixed upstream TRL incident. `workspace/trl` is the `trl/` package of
the TRL 0.27.2 wheel (plus its Apache-2.0 `LICENSE`) with this one local
feature, a `reward_num_workers`
field in `GRPOConfig` (`trl/trainer/grpo_config.py`) and a
`GRPOTrainer._call_reward_func_concurrently` method that
`_calculate_rewards` uses when it is above 1 (`trl/trainer/grpo_trainer.py`).
Every other file is byte-identical to the wheel, and the agent-visible
`source_manifest.json` says only that the tree has local modifications.
The method cuts the batch into contiguous slices, submits one reward call
per slice to a `ThreadPoolExecutor`, and builds the reward list with

```python
for future in as_completed(futures):
    rewards.extend(future.result())
```

`as_completed` yields futures in the order they finish, not the order they
were submitted. `_calculate_rewards` then writes `rewards[i]` into row `i`
of `rewards_per_func`, which is taken to belong to `completions[i]`.
`train_grpo.py` sets `reward_num_workers=32`, which is one slice per
completion at 32 completions per step, and its checker blocks on a child
process. So any score can land on any completion of the step, inside its own
group or in one of the other three.

The user code is correct. `correctness_reward` scores each completion in
order, `check_answer.py` is right for every completion, and the defect is
in a library tree that looks like stock TRL. The method has a docstring and
handles per-sample kwargs carefully. Nothing fails or warns. The process
exits 0, and the logged batch `reward` is still the true mean reward of the
batch, because the right set of scores is only permuted. The run also does
not look like an obvious reward bug. Batch reward starts at 0.07 and decays,
`frac_reward_zero_std` climbs to 0.99 and `grad_norm` to 0. That reads like
"the base model rarely produces the `#### <number>` format and GRPO has no
signal", "the reward is too sparse", "the learning rate or `beta=0` let the
policy drift", or "TRL's importance-sampling correction
(`sampling/importance_sampling_ratio/mean` 0.11-0.17) is shrinking the
updates". None of these is the cause. With the same settings and the scores
collected in submission order, the batch reward averages 0.47 over steps
1-25 and 0.76-0.85 in every later block.

The rollout log does not hand over the answer either. `results/rollouts.jsonl`
logs every fourth completion with the prompt text, the completion, the
reward and the advantage the trainer used, but not the gold answer.
Showing that the rewards are misattached takes one of three routes: join
each prompt back to its GSM8K question to re-check the completion, notice
reward-1 completions that have no `####` line at all, or read the vendored
trainer with `as_completed` semantics in mind. Re-checked, 33 of the 1,600
logged completions are correct and 33 carry reward 1, but only one
completion is both. 32 correct completions carry reward 0 and 32 wrong ones
carry reward 1. 31 of the 33 reward-1 completions contain no `####` at all.

This differs from `reward/verl_math_verify_thread_zero_reward`, where a
grader fails inside worker
threads and scores a subset 0. Here every score is computed correctly and
only its position is wrong, and the reordering is in the framework, not in
task code.

Shipped run (`workspace/results`, seed 20261121, 12 cores), 25-step blocks.
The last three rows come from re-checking the 200 logged completions per
block against the gold answers.

| steps | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch reward | 0.066 | 0.025 | 0.011 | 0.014 | 0.009 | 0.009 | 0.003 | 0.001 |
| `frac_reward_zero_std` | 0.61 | 0.83 | 0.91 | 0.90 | 0.93 | 0.93 | 0.98 | 0.99 |
| `grad_norm` | 0.0035 | 0.0017 | 0.0006 | 0.0010 | 0.0008 | 0.0004 | 0.0002 | 0.0000 |
| logged completions that are correct (of 200) | 14 | 5 | 4 | 4 | 2 | 2 | 2 | 0 |
| P(reward 1 \| correct) | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - |
| P(reward 1 \| wrong) | 0.07 | 0.02 | 0.03 | 0.03 | 0.01 | 0.01 | 0.01 | 0.00 |

The control (same seed and workspace, the diff below applied; recorded, not
shipped) has batch reward 0.469, 0.790, 0.776, 0.846, 0.776, 0.770, 0.836
and 0.761 in the same blocks. Its 1,600 logged rewards all agree with the
re-checked correctness (1,225 correct, all with reward 1). Its
`importance_sampling_ratio/mean` is 0.14-0.39.

### Run-to-run and host variation

The size of the damage depends on which thread finishes first, so it varies
between runs and with the number of CPU cores the job can use. The shipped
run was designated before any of these runs started and was not picked
from repeats.

- **12 cores (3 buggy runs):** all three end at or near 0. The shipped run
  and the seed-20261120 run never pick up the format (the second one ends
  with batch reward 0.001 and 0.0 test accuracy). The third rises to a batch
  reward of 0.35 at its best block and ends at 0.038 test accuracy, with 87%
  of its held-out outputs cut off at the 1,024-token limit.
- **4 and 32 cores (one run each):** the policy learns part of the format but
  degenerates. On 4 cores, 74% of held-out outputs run to the token limit,
  and test accuracy is 0.201 (0.665 if truncated outputs were credited). On
  32 cores the batch reward reaches 0.49 by the end. The step-200 evaluation
  crashed in `common.normalize` (an `OverflowError` on a number too large for
  a float), and the merged checkpoint had already been deleted, so there is
  no test accuracy for that run.
- **2 and 48 cores (one run each):** more scores stay in place, and the buggy
  run still reaches 0.564 (2 cores) and 0.583 (48 cores), below both
  controls (0.723 and 0.753).

The defect is the same on every host; only how much it costs changes. This
is why the task is issue-only: a repaired rerun on an unknown host could not
be told apart reliably from an unrepaired one by the final metric alone.

### All recorded runs

Greedy decoding on all 1,319 GSM8K test items. An output cut off at the
1,024-token limit counts as wrong. "cores" is the CPU count of the job. All
runs used one H100 NVL (94 GB) and 200 steps. Only the shipped run can be
re-checked from this package.

| run | code | seed | cores | step 0 | step 200 | `####` answer at step 200 | truncated at step 200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **shipped evidence** | as shipped | 20261121 | 12 | 0.0167 | **0.0000** | 0.000 | 0.5% |
| buggy | as shipped | 20261120 | 12 | 0.0174 | 0.0000 | 0.000 | 0.4% |
| buggy | as shipped | 20261121 | 12 | 0.0174 | 0.0379 | 0.181 | 87.4% |
| buggy | as shipped | 20261121 | 4 | 0.0182 | 0.2009 | 0.998 | 74.1% |
| buggy | as shipped | 20261121 | 32 | 0.0167 | eval crashed | - | - |
| buggy | as shipped | 20261121 | **2** | 0.0182 | 0.5641 | 0.738 | 2.2% |
| buggy | as shipped | 20261121 | **48** | 0.0167 | 0.5830 | 0.772 | 1.1% |
| **control** | diff below | 20261121 | 12 | 0.0190 | **0.7528** | 0.996 | 0.3% |
| control | diff below | 20261120 | 12 | 0.0182 | 0.7233 | 0.994 | 0.8% |

The step-0 values differ slightly between runs of the same base model
(0.0167-0.0190). vLLM greedy decoding with batched requests is not bitwise
deterministic across processes.

The repository's startup check (`python3 -m autorlbench rerun --backend local`,
run inside the shared image on 1 × L40S with 12 cores) completed on this
package with `passed=true`. It reproduced the failure: 0.0152 at step 0 and
0.0000 at step 200, with no `####` line in any held-out output.

## Reference solution

- **issue**: a passing diagnosis names `as_completed` in the vendored
  `GRPOTrainer._call_reward_func_concurrently` (enabled by
  `reward_num_workers=32`) as the cause. It must say that the slices' scores
  are concatenated in finish order, so `rewards[i]` is no longer the score
  of `completions[i]`, and that GRPO's group-relative advantage then pushes
  up completions that were wrong and pushes down completions that were
  right. It supports this with the reward/correctness disagreement in
  `rollouts.jsonl` (or the code path) and with held-out accuracy staying at
  about 0 while a correctly wired run reaches about 0.75.
- Attributions that must fail:
  - the user's reward function, checker or answer format;
  - learning rate, `beta`, temperature, LoRA or loss settings;
  - reward sparsity or task difficulty;
  - TRL's vLLM importance-sampling correction;
  - the evaluation script;
  - "training did not run".
- **fix** (not graded): keep submission order, as in the diff below.
  `executor.map(lambda b: _call(*b), bounds)` or `reward_num_workers=1` give
  the same rewards.

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -24,7 +24,7 @@
 import warnings
 from collections import defaultdict, deque
 from collections.abc import Callable
-from concurrent.futures import ThreadPoolExecutor, as_completed
+from concurrent.futures import ThreadPoolExecutor
 from contextlib import nullcontext
 from functools import partial
 from pathlib import Path
@@ -1222,7 +1222,7 @@
         rewards = []
         with ThreadPoolExecutor(max_workers=num_workers) as executor:
             futures = [executor.submit(_call, start, end) for start, end in bounds]
-            for future in as_completed(futures):
+            for future in futures:
                 rewards.extend(future.result())
         return rewards
 
```

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing is a conjunctive gate: `verdict_score == 1.0`,
  `root_cause_score > 0.8` and `evidence_score > 0.8`. Issue evaluation
  launches no training and needs no GPU.
- **no solution level**: how many scores `as_completed` moves depends on the
  host's CPU count, and the buggy runs on 2 and 48 cores still reach 0.56 and
  0.58. That is below both controls, but by only 0.14-0.19 from single runs,
  too thin for a calibrated threshold across unknown hosts. `run_spec.json`
  is kept so the harness can reproduce the recorded run.
