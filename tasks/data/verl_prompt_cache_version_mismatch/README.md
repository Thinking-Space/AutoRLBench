# Stale prompt cache misaligns training problems and answers

GRPO training of Qwen/Qwen3-1.7B on DAPO reads cached problem messages from an earlier dataset row order while scoring against current-row answers, leaving clean AIME24/25 mean@32 at 4.48%/3.75% instead of 15.73%/10.31% after 60 steps.

| | |
| --- | --- |
| **Author** | lipengyi0829 |
| **Domain** | data |
| **Tags** | `verl` `grpo` `prompt-cache` `data-versioning` `60-step-budget` `recorded-run` `container-validation-pending` |
| **Model** | Qwen/Qwen3-1.7B |
| **Dataset** | DAPO-Math-17k (1,791,700 rows, duplicates retained); AIME24/25 (30 questions each) |
| **Framework** | verl v0.6.1, `run.sh` |
| **Environment** | `verl-v061-grpo` · 8 GPUs; historical topology: 2 physical workers × 4 A100 |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control run; retrospective steps 0–60 from originally 100-step recipes; one seed per side |

## Difficulty

`case/dataset_cache.py::cache_is_valid(metadata, expected)` treats an unchanged logical source name and row count as sufficient to reuse processed messages. The prepared train parquet contains intact problem/answer pairs after a seeded permutation of complete rows; the seed cache retains the earlier order. `PromptDataset` takes problem messages from that positional cache while the reward reads the current parquet row's answer. Everything remains well formed, so the training process runs without an alignment exception. This is a controlled data-pipeline fault injection, not a claim of an upstream veRL defect.

The strongest attribution evidence is the source/cache content-identity difference together with the loader's positional message lookup and unchanged current-row answer. The staged cache inventory, metadata and problem samples allow that association to be inspected. AIME24/25 bypass this training cache. Both sides use the same V2 data order, Qwen3-1.7B with thinking disabled, batch 512, 8 responses per prompt, PPO mini-batch 32, LR 1e-6, KL disabled and 60-step comparison horizon. The defective run still improves from 1.77%/0.21% at step 0; “no learning” or “below initialization” is not the defect.

| Step 60 | Training reward | AIME24 mean@32 | AIME25 mean@32 | Mean of the two accuracy values |
| --- | ---: | ---: | ---: | ---: |
| Defective cache guard | 0.011962890625 | 4.4792% | 3.7500% | 4.1146% |
| Repaired cache guard | 0.359619140625 | 15.7292% | 10.3125% | 13.0208% |

## Reference solution

- **issue:** identify the stale positional prompt cache, the insufficient source-name/row-count validity check, and the causal chain from V1 messages plus V2 answers to incorrect training feedback. Concrete evidence should connect the cache/source identities, loader code and current-row scoring. Dataset reordering itself preserves valid pairs; reward broadcasting, an intrinsically broken reward formula, AIME label corruption and complete absence of learning are wrong attributions.
- **solution:** bind reusable processed messages to the current source-content SHA256 and preprocessing identity as well as the logical source and row count. On identity mismatch, rebuild the private active cache from the current parquet before tokenization. Keep the dataset, model, scorer, GRPO objective and declared training/evaluation budget unchanged. The canonical patch and control-side records are sealed under `evaluations/solution/oracle/`.

## Verification

- **issue:** the unchanged official LLM judge compares `agent_answer.md` against the hidden oracle. It reports weights verdict/root cause/evidence = 0.2/0.6/0.2. The current implementation passes only when verdict equals 1.0 and both root-cause and evidence scores are strictly greater than 0.8; the weighted score alone is not the pass condition.
- **solution:** the agent repairs the workspace, submits `report.md`, and does not launch training. The harness runs `bash run.sh` for 60 trainer/global steps, evaluates initially and every 10 steps, and grades only the fresh step-60 output under `$AUTORLBENCH_RESULTS/experiment_logs`. The metric is the arithmetic mean of independently rescored AIME24 and AIME25 mean@32, each 30 questions × 32 responses. The proposed final-score interval is `[0.085, 1.0]`. At 60 steps the defective value is 0.0411458333333 (margin 0.0438541666667 below the threshold), and the control value is 0.1302083333333 (margin 0.0452083333333 above it). The threshold also exceeds the control's initial 0.0166666666667. Calibration uses **one run per side**, so these are margins, not variance estimates.

The recorded evidence is a retrospective prefix of the original 100-step runs, not a fresh independently completed 60-step training run. The original save policy kept only step-100 weights; the new 60-step recipe saves its final HF model at step 60. Historical records use SwanLab and JSONL; no offline W&B binary exists. The environment image is defined but **not yet built/validated for this submission**. Container cold/warm preparation, fresh 60-step GPU reruns, and single-host/two-logical-node equivalence remain unverified; the recorded topology was two physical 4-GPU workers. Validator and historical replay receipts should be read separately from fresh-training claims.

Completed checks: the official task validator reports zero issues; 29 offline data/grader tests pass; both historical grader replays separate at the proposed threshold. On the existing Linux training environment, CPU cold/warm derivation recreated all 1,791,700-row Parquet hashes, Hydra composed the 60-step recipe, and the actual native Dataset/tokenizer probe saw 32/32 training-message mismatches on the defective side versus 0/32 after the patch, with clean validation and unchanged current-row labels. These checks did not start GPU training or validate the container image. Receipts are under `reviewer_evidence/checks/`.
