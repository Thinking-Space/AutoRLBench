# Reference repair

`workspace/trl/trainer/grpo_trainer.py`,
`GRPOTrainer._sort_by_completion_length`: the length sort reorders only
tensors with `dim() >= 2`, so the `(B,)` `advantages` tensor stays in the
pre-sort order and each completion is trained with another completion's
advantage. Reorder `advantages` with the rest of the batch.

Apply `fix.diff` from the workspace root (`patch -p1 < fix.diff` in
`workspace/`). It changes the filter to `value.dim() >= 1`. Indexing
`advantages` explicitly, or removing the `_sort_by_completion_length` call
from `_prepare_inputs`, gives the same alignment; `num_items_in_batch` is 0-d
and is not affected by either filter.

## Recorded comparison

All runs used one H100 NVL, 12 CPU cores, 100 steps, and the shared
`vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` image through Apptainer.

| arm | seed 20261120 | seed 20261121 | seed 20261122 | seed 20261123 | seed 20261124 | seed 20261125 |
| --- | --- | --- | --- | --- | --- | --- |
| as shipped | **0.0000** (shipped) | 0.4268 | 0.0000 | 0.0083 | 0.2691 | 0.0045 |
| `fix.diff` | **0.7384** | 0.7468 | 0.7210 | | | |

Values are the grader's re-score of each run's `eval_after.json.items.jsonl`,
which equals the logged `eval/gsm8k_accuracy` for every run. The threshold
0.60 is 0.17 above the best buggy run and 0.12 below the lowest control.

`recorded_control/experiment_logs` holds the seed-20261120 control's
`metrics.jsonl`, evaluation summaries, `eval_after.log` and the scored dump,
enough to replay the grader offline. Its batch reward is 0.350, 0.736, 0.780,
0.788 and 0.845 in 20-step blocks.
