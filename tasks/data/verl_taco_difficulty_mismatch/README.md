# Training data is too hard

Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set is very hard while the validation set is easy, so held-out accuracy ends below where it started.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | data |
| **Tags** | `verl` `grpo` `taco` `data-difficulty` `issue-only` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO `VERY_HARD` train (2329 rows) · TACO `EASY` validation (184 rows) |
| **Framework** | verl, `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh` |
| **Environment** | `verl:vllm017.latest` · no GPU (issue-only, no training is started) |
| **Evaluations** | issue |
| **Recorded runs** | 1 run, 400/400 steps, ~8 h 41 min on 8 GPUs |

## Difficulty

There is no code defect. The code, launch hyperparameters, and model are a healthy baseline. The failure lives in data selection: `train.parquet` holds only TACO `VERY_HARD` rows and `test.parquet` is the existing `EASY` split.

The signals point in opposite directions. Training reward climbs from ~0.008 to a stable ~0.15 after step 200. Held-out accuracy peaks at 0.0978 (step 50) and ends at 0.0380 (7/184), below step 0 (0.0489). The drop is small enough to pass for noise. Reaching the cause means comparing the difficulty makeup of train and validation, not reading the code.

## Reference solution

- **issue**: either "the training set is too hard" or "train and validation difficulty do not match" counts as the full root cause. No code defect is required.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.

Known limitation: the task directory name and the published dataset repo name both describe the answer, and the agent can see both.
