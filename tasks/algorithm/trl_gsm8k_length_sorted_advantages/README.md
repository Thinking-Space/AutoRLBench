# Length-sorted micro-batches leave GRPO advantages behind

Qwen2.5-1.5B-Instruct is trained with TRL GRPO on GSM8K. The workspace's
vendored TRL sorts each generation batch by completion length before cutting
it into gradient-accumulation micro-batches, so that each micro-batch can drop
the padding it does not use. The sort reorders every two-dimensional tensor of
the batch but passes the one-dimensional `advantages` through unchanged, so
from then on each completion is trained with another completion's advantage.
Rewards, advantages and the rollout log are all computed and recorded
correctly before this point. The batch reward drifts from 0.10 to 0.03 and
greedy GSM8K test accuracy goes from 0.014 to 0.000 after 100 steps. The same
workspace with the advantages sorted along with the rest reaches 0.74.

| | |
| --- | --- |
| **Author** | Yaolun Zhang |
| **Affiliation** | Oregon State University |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `advantage-misalignment` `micro-batching` `vendored-framework` `silent-failure` |
| **Framework** | TRL 0.27.2, vendored in `workspace/trl` with local patches (GRPO, `dr_grpo` loss, colocated vLLM 0.11.0); transformers 4.57.6, peft 0.19.1 |
| **Run time** | ~9 minutes on an H100 NVL (two 1,319-item evaluations plus 100 training steps) |
| **Resources** | 1 × H100 NVL (94 GB), 12 CPU cores |

## Difficulty

`workspace/trl` is the `trl/` package of the TRL 0.27.2 wheel (plus its
Apache-2.0 `LICENSE`) with two local changes to `trl/trainer/grpo_trainer.py`
and one to `trl/trainer/grpo_config.py`. The agent-visible
`source_manifest.json` says only that the tree has local modifications.

The defect is in `GRPOTrainer._sort_by_completion_length`, which
`_prepare_inputs` calls between TRL's own `shuffle_sequence_dict` and
`split_tensor_dict`:

```python
lengths = batch["completion_mask"].sum(dim=1)
order = torch.argsort(lengths, stable=True)
sorted_batch = {}
for key, value in batch.items():
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        value = value[order]
    sorted_batch[key] = value
```

The docstring describes the intent: reorder the "row-aligned token tensors",
pass "batch-level entries such as `num_items_in_batch`" through. The filter
`dim() >= 2` encodes that, and it misses that GRPO's `advantages` is a
per-sample tensor of shape `(B,)`. Prompt and completion ids, masks, old and
sampling log-probabilities and the `(B, 1)` sequence-level IS ratios move
together; `advantages` stays in the shuffled order. `compute_loss` then
multiplies the log-probabilities of the completion in row `i` by
`advantages[i]`, which belongs to whichever completion sat in row `i` before
the sort. The companion `_trim_completion_padding` is correct, and the loss
normalizer (`dr_grpo` divides by `max_completion_length`) does not depend on
the trimmed width.

Several things keep this hidden:

- **Every log is correct.** `results/rollouts.jsonl` and the
  `reward`/`reward_std`/`frac_reward_zero_std` metrics are written in
  `_generate_and_score_completions`, before the sort. Re-checked against the
  gold answers, all 800 logged completions carry the reward they earned, and
  each logged advantage is the one the trainer computed for that completion
  from its group. A diagnosis that audits the reward pipeline finds nothing
  wrong.
- **A second local feature looks more suspicious and is correct.**
  `train_grpo.py` sets `reward_num_workers=32`, a local `GRPOConfig` field
  that scores one-completion slices from a thread pool, and its checker runs
  in a child process. That code concatenates the slices in submission order
  and is correct, as the rollout log shows.
- **The run looks like ordinary GRPO trouble.** The base model writes the
  required `#### <number>` line on only 2% of test items. The batch reward
  starts at 0.10 and decays, `frac_reward_zero_std` climbs from 0.46 to about
  0.8, and by step 100 6% of training completions and 28% of held-out outputs
  run to the 1,024-token limit. That reads like "the reward is too sparse",
  "the learning rate or `beta=0` lets the policy drift", or "TRL's vLLM
  importance-sampling correction (`importance_sampling_ratio/mean` 0.11-0.16)
  shrinks the updates". None of these is the cause. The in-order control has
  the same sparsity and the same IS correction and climbs to batch reward
  0.74 by steps 21-40.
- **Nothing fails.** The process exits 0 and nothing is logged as an error or
  warning. `grad_norm` stays non-zero (0.002-0.005), so training is
  happening.

Finding it takes reading the vendored training path past the reward and
advantage computation, as far as the step where the generation batch is
split into micro-batches, and checking the shape of every entry the sort
leaves alone. The logs can only show that a correctly rewarded,
correctly normalized signal was present and the policy did not follow it.

Shipped run (`workspace/results`, seed 20261120 = the default in
`run_grpo.sh`, 12 cores), 20-step blocks:

| steps | 1-20 | 21-40 | 41-60 | 61-80 | 81-100 |
| --- | --- | --- | --- | --- | --- |
| batch reward | 0.102 | 0.036 | 0.017 | 0.044 | 0.028 |
| `frac_reward_zero_std` | 0.46 | 0.75 | 0.86 | 0.73 | 0.83 |
| `grad_norm` | 0.0046 | 0.0021 | 0.0026 | 0.0032 | 0.0042 |
| completions cut at 1,024 tokens | 0.0% | 0.0% | 0.2% | 0.5% | 5.9% |
| logged completions that are correct (of 160) | 16 | 4 | 4 | 5 | 3 |
| of those, logged with reward 1 | 16 | 4 | 4 | 5 | 3 |

Held-out: 0.0144 → **0.0000** test accuracy; outputs with a `####` line 2.1%
→ 0.0%; outputs cut off at the token limit 0.4% → 28.0%.

The control (same seed and workspace with the diff below; recorded, not
shipped) has batch reward 0.350, 0.736, 0.780, 0.788 and 0.845 in the same
blocks and ends at 0.7384 test accuracy.

### Run-to-run variation

With the advantages detached from their completions, each update raises or
lowers the probability of sampled completions without regard to whether they
were right. Where the policy drifts differs between runs. In four of six
100-step runs the policy never picks up the format (0.000-0.008 test
accuracy). In the other two it drifts onto the `####` format and the base
model's arithmetic carries it to 0.27 and 0.43, still well below every
control (0.72-0.75). The shipped run is the default seed and was designated
before any run finished.

All runs: greedy decoding on all 1,319 GSM8K test items, an output cut off at
the 1,024-token limit counts as wrong, one H100 NVL, 12 CPU cores unless
noted. Only the shipped run can be re-checked from this package.

| run | code | seed | steps | step 0 | final | `####` line at end | truncated at end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **shipped evidence** | as shipped | 20261120 | 100 | 0.0144 | **0.0000** | 0.000 | 28.0% |
| buggy | as shipped | 20261121 | 100 | 0.0152 | 0.4268 | 0.539 | 0.9% |
| buggy | as shipped | 20261122 | 100 | 0.0159 | 0.0000 | 0.000 | 0.2% |
| buggy | as shipped | 20261123 | 100 | 0.0144 | 0.0083 | 0.012 | 16.9% |
| buggy | as shipped | 20261124 | 100 | 0.0167 | 0.2691 | 0.365 | 0.3% |
| buggy | as shipped | 20261125 | 100 | 0.0167 | 0.0045 | 0.008 | 0.6% |
| **control** | diff below | 20261120 | 100 | 0.0144 | **0.7384** | 0.989 | 0.4% |
| control | diff below | 20261121 | 100 | 0.0159 | 0.7468 | 0.995 | 0.4% |
| control | diff below | 20261122 | 100 | 0.0152 | 0.7210 | 0.995 | 0.2% |
| buggy (200 steps) | as shipped | 20261121 | 200 | 0.0174 | 0.0008 | 0.001 | 0.0% |
| buggy (200 steps) | as shipped | 20261120 | 200 | 0.0144 | 0.6126 | 0.847 | 0.5% |
| buggy (200 steps) | as shipped | 20261122 | 200 | 0.0152 | 0.0000 | 0.000 | 1.1% |
| buggy (200 steps, 4 cores) | as shipped | 20261121 | 200 | 0.0144 | 0.0091 | 0.013 | 0.5% |
| control (200 steps) | diff below | 20261121 | 200 | 0.0174 | 0.7521 | 0.997 | 0.1% |
| control (200 steps) | diff below | 20261120 | 200 | 0.0152 | 0.6748 | 0.998 | 0.7% |

The 200-step runs came first. One buggy 200-step run found the format late
(batch reward 0.38 and 0.67 in its last two 25-step blocks), while the
controls pass 0.78 by steps 26-50, so the task uses 100 steps. The step-0
values differ slightly between runs of the same base model (0.0144-0.0174)
because vLLM greedy decoding with batched requests is not bitwise
deterministic across processes.

The repository's startup check (`python3 -m autorlbench rerun --backend local`,
run inside the shared image on 1 × H200 with 12 cores) completed on this
package with `passed=true` in 532 s. It reproduced the failure: 0.0144 at
step 0 and 0.0023 at step 100. The solution grader scores that rerun's logs
0.0023 and fails it.

## Reference solution

- **issue**: a passing diagnosis names `GRPOTrainer._sort_by_completion_length`
  in the vendored `trl/trainer/grpo_trainer.py` as the cause. It must say that
  the sort reorders only tensors with `dim() >= 2`, that `advantages` has
  shape `(B,)` and is left in the pre-sort order, and that `compute_loss`
  therefore weights each completion with another completion's advantage, so
  the policy gradient no longer points toward correct answers. It supports
  this with the logs showing a correct, non-degenerate reward and advantage
  signal (rollout rewards match correctness, `frac_reward_zero_std` 0.46 at
  the start) that the policy does not follow, and with held-out accuracy
  going from 0.014 to 0.000 while a correctly wired run reaches about 0.74.
- Attributions that must fail:
  - the reward function, the checker, `reward_num_workers` or the thread
    pool;
  - the answer format, reward sparsity or task difficulty;
  - learning rate, `beta`, temperature, LoRA or loss settings;
  - TRL's vLLM importance-sampling correction;
  - truncation or the 1,024-token limit;
  - the evaluation script.
- **solution**: sort `advantages` with the rest of the batch. Changing the
  filter to `value.dim() >= 1` (the diff below), indexing `advantages`
  explicitly, or removing the `_sort_by_completion_length` call all give
  aligned advantages. The recorded controls use the diff below.

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -1195,7 +1195,7 @@
         order = torch.argsort(lengths, stable=True)
         sorted_batch = {}
         for key, value in batch.items():
-            if isinstance(value, torch.Tensor) and value.dim() >= 2:
+            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                 value = value[order]
             sorted_batch[key] = value
         return sorted_batch
```

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing is a conjunctive gate: `verdict_score == 1.0`,
  `root_cause_score > 0.8` and `evidence_score > 0.8`. Issue evaluation
  launches no training and needs no GPU.
- **solution**: the harness reruns `bash run_grpo.sh` (`run_spec.json`:
  seed 20261120, 100 steps) in the repaired workspace. The grader re-scores
  the fresh run's 1,319 held-out outputs (`eval_after.json.items.jsonl`) with
  its own copy of the gold answers. A row counts only if it finished on its
  own and its last `#### <number>` equals the gold answer. Pass requires
  accuracy ≥ **0.60**. That is 0.17 above the best of six buggy 100-step runs
  (0.4268) and 0.12 below the lowest of three controls (0.7210); offline
  replay of the grader fails all six buggy runs and passes all three
  controls. `metrics.jsonl` must show all 100 training steps and the trainer's
  summary row. `run_grpo.sh`, `train_grpo.py`, `eval_gsm8k.py`, `common.py`,
  `check_answer.py` and the two GSM8K parquets are pinned, so the repair has
  to be in `workspace/trl`.
