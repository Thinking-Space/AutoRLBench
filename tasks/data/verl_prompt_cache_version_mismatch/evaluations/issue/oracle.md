# Canonical issue oracle

Reconciled from the controlled source pair and the frozen steps 0–60 from two originally 100-step experiments. There is one essential data-pipeline root cause.

Expected verdict: **unhealthy**.

## Root cause

- `case/dataset_cache.py::cache_is_valid(metadata, expected)` — kind: `symbol`, stale processed-data cache identity.
  - Observed: the guard accepts a cache when `identity.source_key` and `identity.rows` agree with the current source. It does not require the source-content SHA256 or preprocessing identity to agree.
  - Expected: accept processed messages only when the source name, row count, current content SHA256 and preprocessing identity all match. A mismatch must invalidate and rebuild the private active cache from the current train parquet.
  - Causal role: V1 and V2 contain the same intact original DAPO problem/answer rows, but V2 applies a complete-row permutation. The logical `train.parquet` name and row count are unchanged, so the weak guard incorrectly reuses V1 processed messages. `PromptDataset` then takes cached messages by the prepared position while keeping the answer and reward fields from the V2 row. The model receives an earlier problem and is scored against another problem's answer. GRPO is updated using semantically incorrect per-response feedback even though the reward function computes correctness as implemented.

The complete-row permutation itself is not a corrupt-data defect: each V2 parquet row remains correctly paired before the cache is applied. The defect is the acceptance and reuse of processed messages from a different data revision.

## Observable impact

- The main paired held-out measure is clean AIME24/25 mean@32 at step 60, with 30 questions and 960 sampled responses per year; both arms use the same held-out questions and gold answers.
- Defective run at step 60: training reward `0.011962890625`; AIME24 `0.04479166666666667` (43/960); AIME25 `0.0375` (36/960). The two-year mean is `0.04114583333333333`.
- Control at step 60: training reward `0.359619140625`; AIME24 `0.15729166666666666` (151/960); AIME25 `0.103125` (99/960). The two-year mean is `0.13020833333333334`.
- The cache repair improves AIME24 by 11.25 percentage points and AIME25 by 6.5625 percentage points. These are observed single-run margins, not a multi-seed variance claim.
- The defective run's own step-0 values are `0.017708333333333333` and `0.0020833333333333333`. Its step-60 held-out scores are still above initialization. Its validation rises earlier and falls after step 40; do not attribute this case as complete absence of learning or as already below initialization at step 60.
- The frozen `outputs/metrics.jsonl`, `outputs/train.log`, `outputs/historical_config.yaml` and `outputs/frozen_eval/60.jsonl` provide metric, configuration and answer evidence. `outputs/cache_inventory.json`, prepared `data/manifest.json` and seed-cache metadata provide identities and association evidence. Historical labels/paths may be sanitized without changing metric values.

These observations are symptoms and supporting evidence, not additional root causes. Training reward is itself computed against the mismatched training labels, so clean held-out evaluation is the capability measure.

## Trigger conditions

- A processed structured-message cache was built from an earlier train-data revision.
- The current revision permutes whole rows while preserving the logical source name and total number of rows.
- The runtime guard checks only that stable name and count, then retrieves cached messages by positional index.
- The reward continues to use the current row's unchanged answer. Both tokenized inputs and async rollout prompt IDs are derived from the effective cached messages.
- Validation bypasses this training cache, so normal AIME scoring exposes the impaired learning rather than inheriting the same prompt/answer misassociation.

## Documentation reconciliation

- This is a controlled fault injection in a custom data loader, not evidence that native veRL 0.6.1 necessarily ships this bug.
- Both arms use Qwen3-1.7B with thinking disabled, healthy independent correctness rewards, standard GRPO, KL disabled, the same V2 rows, 2048-token prompt cap, 8192-token response cap, padding removal and dynamic batches. A diagnosis must connect an actual data-flow error to the observed behavior; naming an unusual but shared setting is insufficient.
- Do not confuse the bug with the earlier batch-reward-broadcast case. Here individual answers are scored separately; the wrong problem/answer association enters before generation.
- Do not blame the V2 parquet's intact rows, duplicated original DAPO records, validation labels or a changed scoring formula. Do not propose truncation, model size, KL settings, learning rate, more training, or synthetic answer contamination as this case's demonstrated cause.
- A statement that the run learned somewhat is compatible with the expected unhealthy verdict. A statement that the run is healthy solely because some held-out improvement occurred misses the cache misassociation and the matched repair gap.
- The selected horizon is a retrospective 60-step prefix. No step-60 model checkpoint was saved in the original run; the stored validation answers, rather than newly generated model answers, support this historical comparison.
