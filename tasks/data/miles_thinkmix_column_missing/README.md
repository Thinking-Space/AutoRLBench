# Training parquet has no think_mode column; THINK_MODE=auto silently falls back to full-think

Training Qwen3.5-9B with arena-rubric GRPO using miles. The training parquet was
never processed by `add_thinkmix.py` and therefore has no `think_mode` column.
With `THINK_MODE=auto`, the miles framework logs a warning and silently trains
100% think samples instead of the intended 70/30 think/nothink split.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Data |
| **Domain** | Arena-Rubric RL |
| **Tags** | `thinkmix` `think_mode` `silent-fallback` `parquet-schema` `add_thinkmix` |
| **Framework** | miles (GRPO) |
| **Run time** | ~13 hours (200 rollouts) |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers. The agent sees only `workspace/`
and the evaluation's `instruction.md`.

Evidence is reconstructed from session documentation. The defect is real;
the parquet schema and framework warning are faithful reproductions.

## Difficulty

The defect is in the data pipeline: `add_thinkmix.py` was never run on the training
parquet, so the file has schema `[prompt, data_source, ability, reward_model, extra_info]`
— no `think_mode` column. The launch script sets `THINK_MODE=auto` and
`--think-mode-key think_mode`, which instructs the miles framework to read
per-sample think/nothink assignments from the `think_mode` column.

When the column is absent, the framework logs a single warning line:
```
[warn] think_mode column not found in dataset; falling back to full think mode
```
and continues training with 100% think samples. No error is raised, training
runs to completion, and the checkpoint saves normally. The only observable
symptoms are:
- `arena_rubric/think_ratio` is 1.0 (should be ~0.70)
- `arena_rubric/nothink_ratio` is 0.0 (should be ~0.30)
- Nothink evaluation scores remain at base-model level

Nothing in the launch log or the training reward curve reveals the problem.
The filename `tigan_10k_rubric_G_argon_improved_q359b_thinkmix73_seed42.parquet`
contains `thinkmix73` as a naming artifact from copying a previous spec; the file
was never actually processed.

The fix is to verify the parquet schema before launching training:
```python
import pyarrow.parquet as pq
schema = pq.ParquetFile(path).schema_arrow
assert 'think_mode' in schema.names, f"think_mode column missing from {path}"
```

## Reference solution

- **issue**: identify that the training parquet has no `think_mode` column, that
  `THINK_MODE=auto` causes miles to silently fall back to full-think, and connect
  this to `think_ratio=1.0` and flat nothink evaluation scores. Reject "THINK_MODE
  is misconfigured" (it is correctly set) and "the model learned think-mode only"
  as root cause explanations without identifying the missing column.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; the fix requires re-running data preparation
and a full training run, which produces a qualitatively different training distribution
not suitable for single-threshold grading.
