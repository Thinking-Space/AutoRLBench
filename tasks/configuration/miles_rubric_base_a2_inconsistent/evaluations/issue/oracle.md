# Canonical issue oracle

Reconciled from session-documented evidence.
Expected verdict: **unhealthy**.

## Root cause

- `gen_rubric_tigan10k.py::A2_PARQUET` module-level constant; `--build` invocation env
  - kind: `configuration`
  - observed: `gen_rubric_tigan10k.py` reads `BASE_A2` at two distinct call sites:
    (1) during generation, to populate the a2 answer slot that shapes rubric criteria;
    (2) during `--build`, inside `load_answers()`, to write `reward_model.reference`
    into the training parquet. The `--build` invocation was called without `BASE_A2`
    set, so `load_answers()` silently fell back to the module-level default
    `A2_PARQUET = "data/answers_moebase.parquet"` (the old base model answers).
    The resulting parquet has rubric criteria calibrated against 9B answers but
    `reward_model.reference` containing old-base answers.
  - expected: the same `BASE_A2` pointing to the 9B answer parquet is passed to
    both invocations (generation and `--build`), so criteria and reference match.

## Observable impact

- Primary symptom: when training reward is decomposed, rubric criteria consistently
  favor long answers calibrated against the 9B model, but the reference answer slot
  contains the old base model's shorter outputs. Reward is systematically misaligned.
- Detectable by sampling 50+ uid pairs from the parquet and comparing the
  `reward_model.reference` text to the `data/answers_q359b.parquet` answers for
  the same uid — they will differ.
- Training runs to completion with no error.

## Trigger conditions

- `gen_rubric_tigan10k.py --build` called without `BASE_A2` in the environment.
- `load_answers()` reads the module-level default when `BASE_A2` is not set.

## Documentation reconciliation

- "The generation was correct" is true but incomplete: the rubric criteria
  were generated correctly with 9B answers; the mismatch is in the `--build` step.
- "The reward is low because the model hasn't learned yet" is a symptom; the
  root cause is that the reference answers the reward function compares against
  are from the wrong model.
- Both `data/answers_q359b.parquet` and `data/answers_moebase.parquet` exist in
  the workspace — the agent should diff their content for the same uid to confirm.
