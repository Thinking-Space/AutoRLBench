# Canonical issue oracle

Reconciled from session-documented evidence (Qwen3.5-9B arena-rubric RL;
evidence is a faithful synthetic reconstruction from documented framework behavior).

Expected verdict: **unhealthy**.

## Root cause

- `run-qwen3_5-9b-arena-rubric.sh::MEGATRON_MODEL_PATH`
  - kind: `hparam`
  - observed: `MEGATRON_MODEL_PATH` is set to `<base>/Qwen3.5-9B_torch_dist/release`
    — the `release/` subdirectory that holds the actual checkpoint shards.
    Megatron's `checkpointing.py` looks for `latest_checkpointed_iteration.txt`
    in `MEGATRON_MODEL_PATH` to determine which iteration to load. That file
    lives in `<base>/Qwen3.5-9B_torch_dist` (the parent directory), not in
    `release/`. When it is absent, Megatron logs exactly:

    ```
    will not load any checkpoints and will start from random
    ```

    and then continues. All subsequent rollouts are generated from a randomly-
    initialized model; reward stays near zero throughout all training steps.
  - expected: `MEGATRON_MODEL_PATH` is set to `<base>/Qwen3.5-9B_torch_dist`
    (the parent directory containing `latest_checkpointed_iteration.txt`), so
    Megatron successfully loads the checkpoint before training starts.

## Observable impact

- Primary source metric: training reward. With a randomly-initialized model,
  `arena_rubric/score_mean` stays near 0 from step 0 through the end of training;
  there is no warm-start reward bump.
- The driver log contains the line:
  ```
  will not load any checkpoints and will start from random
  ```
  within the first 200 lines of training output.
- Training completes without error; checkpoints are saved normally.
- SGLang rollout responses are semantically incoherent (random token sequences)
  since weights are synchronized from the randomly-initialized Megatron model.

## Trigger conditions

- `MEGATRON_MODEL_PATH` points one level too deep: to `release/` rather than
  the parent directory that holds `latest_checkpointed_iteration.txt`.

## Documentation reconciliation

- "The rubric judge failed" is a wrong attribution: the random model produces
  syntactically valid responses that the judge scores, but scores are uniformly
  low because the responses are incoherent. The judge itself is functioning.
- "The model did not converge" is a symptom, not a root cause.
- "The checkpoint path exists" is true but misleading: the directory
  `release/` exists; the missing file is `latest_checkpointed_iteration.txt`
  one level up.
- The fail-fast check (`if [ ! -f "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt" ]`)
  that the fixed workspace adds is not present in the defective workspace.
