# Canonical issue oracle

Reconciled from session-documented evidence (Qwen3.5-9B arena-rubric RL training;
evidence is a faithful synthetic reconstruction from documented framework behavior).

Expected verdict: **unhealthy**.

## Root cause

- `run-qwen3_5-9b-arena-rubric.sh` and the training parquet file `TRAIN_FILE` points to
  - kind: `data`
  - observed: `TRAIN_FILE` resolves to a parquet with schema
    `[prompt, data_source, ability, reward_model, extra_info]` — no `think_mode`
    column. `add_thinkmix.py` was never run on this file. `THINK_MODE=auto` instructs
    the miles framework to read per-sample think/nothink assignments from the
    `think_mode` column; when the column is absent, the framework logs:
    ```
    [warn] think_mode column not found in dataset; falling back to full think mode
    ```
    and trains 100% think samples for all 200 rollouts.
  - expected: `TRAIN_FILE` resolves to a parquet that has been processed by
    `add_thinkmix.py` and contains a `think_mode` column with values `think` and
    `nothink` in approximately 70/30 proportion; the framework then trains the
    intended mixed distribution and both `think_ratio` and `nothink_ratio` are
    non-zero.

## Observable impact

- `arena_rubric/think_ratio` metric stays at 1.0 for all 200 rollouts.
- `arena_rubric/nothink_ratio` metric stays at 0.0 for all 200 rollouts.
- Nothink evaluation scores remain at base-model level because the policy never
  receives nothink training signal.
- Training runs to completion with no error or crash; the checkpoint saves
  normally.

## Trigger conditions

- `THINK_MODE=auto` is set in the launch config.
- `TRAIN_FILE` points to a parquet that was never processed by `add_thinkmix.py`
  and therefore has no `think_mode` column.
- The parquet filename may contain the string `thinkmix73` as a naming artifact
  from copying an earlier spec; the filename alone is not evidence that
  `add_thinkmix.py` was run.

## Documentation reconciliation

- "THINK_MODE=auto is correctly set" is not evidence of health: the config is
  correct, but the data the config acts on lacks the required column.
- "The model learned think-mode only" is a symptom, not a root cause.
- The `thinkmix73` substring in the filename is a naming artifact, not proof
  that `add_thinkmix.py` was applied. The parquet schema is the authoritative
  source.
- Neither the training reward curve nor the validation metrics directly reveal
  the think-ratio issue; the tell is `think_ratio=1.0` and `nothink_ratio=0.0`.
