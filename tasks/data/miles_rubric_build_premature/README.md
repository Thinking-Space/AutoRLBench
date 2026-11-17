# Rubric --build runs before generation completes, producing a parquet with 15% fewer samples

Training Qwen3.5-9B with arena-rubric GRPO using miles. The rubric generation
script `gen_rubric_tigan10k.py` has async generation writing to a JSONL file.
`--build` was called while generation was still running, so `load_done()` read
8199 completed rows and froze the parquet. Generation later completed to 9716
rows, but the parquet was already frozen at 8199 — 15% fewer samples than intended.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Data |
| **Domain** | Arena-Rubric RL |
| **Tags** | `rubric` `premature-build` `data-pipeline` `jsonl` `parquet` `partial-state` |
| **Framework** | miles (GRPO) |
| **Run time** | Data pipeline stage (~4-8 hours generation) |
| **Resources** | 0 GPU (CPU-only data task) |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is a timing issue in the data pipeline. `gen_rubric_tigan10k.py`
supports two modes:
1. Default (no `--build`): runs async rubric generation, appending rows to a JSONL
2. `--build`: reads the JSONL via `load_done()` and builds the training parquet

`load_done()` reads whatever rows are present at call time — it does not check
whether generation is complete. When `--build` is called concurrently with
generation, it freezes a partial state.

The "待生成 0" gate is the canonical completeness check: re-run the script without
`--build`; if it prints "待生成 0" (zero pending), generation is complete and
`--build` is safe. The defective invocation skipped this gate.

Nothing fails loudly: the parquet is structurally valid, training runs to
completion, and the only symptom is that the training dataset has 8199 rows
instead of the expected 9716 — a 15.6% reduction that silently weakens the
training signal.

## Reference solution

- **issue**: identify that `gen_rubric_tigan10k.py --build` was called before
  generation completed, that `load_done()` reads a partial JSONL state without
  a completeness check, and connect this to the 8199 vs 9716 row count
  discrepancy. Reject "the generation script has a bug" (generation completed
  correctly) and "the training data is too small" (symptom, not cause).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; the fix requires re-running the data
pipeline and a full training run.
