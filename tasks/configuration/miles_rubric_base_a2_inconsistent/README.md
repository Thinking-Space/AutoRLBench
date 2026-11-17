# BASE_A2 passed at rubric generation but omitted at build time, mismatching rubric criteria and reference answers

Training Qwen3.5-9B with arena-rubric GRPO using miles. `gen_rubric_tigan10k.py` reads
`BASE_A2` at two separate call sites — once during generation (shaping rubric criteria)
and once during `--build` (writing `reward_model.reference` into the parquet). The
`--build` invocation omitted `BASE_A2`, so `load_answers()` silently used the old base
model answers as reference. Training proceeded with rubric criteria calibrated against
9B answers but `reward_model.reference` containing old-base answers.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Configuration |
| **Domain** | Arena-Rubric RL |
| **Tags** | `BASE_A2` `reference-mismatch` `arena-rubric` `data-pipeline` `silent-failure` |
| **Framework** | miles (GRPO) |
| **Run time** | Data pipeline stage |
| **Resources** | 0 GPU (CPU-only data task) |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

`gen_rubric_tigan10k.py` has two independent call sites that read `BASE_A2`:

1. **Generation mode** (no `--build`): `get_a2_answers()` reads `BASE_A2` to fetch the
   answer set that fills the a2 slot in the rubric prompt. These answers influence which
   rubric criteria the judge generates — criteria calibrated against 9B outputs.

2. **Build mode** (`--build`): `load_answers()` reads `BASE_A2` again to write
   `reward_model.reference` into every parquet row. This reference is what the training
   reward function compares model outputs against.

The module-level fallback is `A2_PARQUET = "data/answers_moebase.parquet"` (the old
base model). When `--build` is called without `BASE_A2` set, `load_answers()` silently
reads this default instead of the 9B answers. No error is raised. The parquet is
structurally valid, training runs to completion, and the reward function operates
normally — it just compares against the wrong reference.

The only way to catch this is to sample 50+ uid pairs from the parquet and compare
`reward_model.reference` text against `data/answers_q359b.parquet` for the same uid.
They will differ: one is the 9B model's answer, the other is the old base model's.

## Reference solution

- **issue**: identify that `load_answers()` has a module-level default `A2_PARQUET`
  pointing to the old base model, that `--build` was called without `BASE_A2` set,
  and that the resulting parquet has `reward_model.reference` from the wrong model.
  Connect this to the two-call-site structure of `gen_rubric_tigan10k.py`. Reject
  "the rubric generation was wrong" (generation ran with the correct BASE_A2) and
  "training reward is too low" (symptom, not cause).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; the fix requires re-running the data pipeline
and a full training run.
