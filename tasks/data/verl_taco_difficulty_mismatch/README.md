# Training data is too hard

Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set holds only
`VERY_HARD` problems, so almost every GRPO group is all-zero and training never
learns; held-out accuracy stays flat.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | data |
| **Tags** | `verl` `grpo` `taco` `data-difficulty` `issue-only` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO `VERY_HARD` train (2329 rows) · TACO `EASY` validation (184 rows) |
| **Framework** | verl, `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh` |
| **Environment** | `verl` · no GPU (issue-only, no training is started) |
| **Evaluations** | issue |
| **Recorded runs** | 1 run, 400/400 steps, ~15 h 18 min on 4 GPUs |

## Difficulty

There is no code defect. The code, launch hyperparameters
(`use_kl_loss=True`, `kl_loss_coef=0.02`, `entropy_coeff=0.001`), and model are
a healthy baseline. The failure lives in data selection: `train.parquet` holds
only TACO `VERY_HARD` rows and `test.parquet` is the existing `EASY` split.

The TACO reward is binary (1 only when every test passes). On the training
problems the policy almost never earns a 1, so nearly every GRPO group of 8
rollouts is all-zero, its group-normalized advantage is exactly 0, and the
policy-gradient term vanishes on most steps. The policy-gradient signal is sparse, and held-out accuracy fluctuates
without a clear improvement over its step-0 level. The train/validation
difficulty mismatch is evidence for "too hard", not the mechanism.

`workspace/rollout_data/` holds the per-step training rollout dumps (400 files
x 128 rollouts) and `workspace/validation_data/` the nine validation dumps (184
rollouts each) — the primary quantitative evidence for the causal chain.

## Recorded run

- Log: `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261021_080557.log`
- Wandb: `workspace/logs/wandb/offline-run-20261021_080907-4pye3rsp`
- 4 GPUs, 400/400 steps in 15:18:01 (137.70 s/step); 145 steps per epoch (2329
  training rows), so the step cap stops training in the third epoch.
- Config: `train_batch_size=16`, `rollout.n=8`, `lr=1e-6`, rollout temperature
  1.0, validation temperature 0.7.

Held-out accuracy on 184 TACO `EASY` rows (counts are `round(acc * 184)`):

| Step | Fully correct | acc |
| --- | ---: | ---: |
| 0 | 13/184 | 0.0707 |
| 50 | 17/184 | 0.0924 |
| 100 | 16/184 | 0.0870 |
| 150 | 16/184 | 0.0870 |
| 200 | 13/184 | 0.0707 |
| 250 | 15/184 | 0.0815 |
| 300 | 13/184 | 0.0707 |
| 350 | 12/184 | 0.0652 |
| 400 | 14/184 | 0.0761 |

Training signal: `critic/rewards/mean` averages 0.0078; 254/400 steps are 0;
0.78% of 51,200 training rollouts succeed; 175/6,400 groups are mixed; entropy
rises 0.69 -> ~1.0-1.1 (a symptom of zero-advantage steps, not a cause).

## Reference solution

- **issue**: state that the training data is too hard for the model and
  connect it to the missing learning signal (near-zero reward, all-zero
  groups / zero advantages, flat held-out curve). Either half alone earns
  partial root-cause credit only.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Pass requires `verdict_score` == 1.0 and `root_cause_score`
  and `evidence_score` each > 0.8. Full root-cause credit requires the causal
  chain; "too hard" alone or a bare difficulty mismatch gets 0.3 and cannot
  pass.

Known limitation: the task directory name describes the answer. The dataset
repo name also states it, but it is base64-encoded inside `prepare.sh` and
cached under a neutral `--cache-as` alias. This hides the literal name from
simple searches but does not prevent disclosure: the agent can read and decode
`prepare.sh`. Base64 is not an access boundary.

Audit note (2026-10-31): all 51,200 training records parse when read as
LF-delimited JSONL. One response in `workspace/rollout_data/380.jsonl` contains
a legal Unicode U+2028 character; `str.splitlines()` incorrectly splits that
record. Group by prompt, not adjacent blocks of eight: the intact evidence
has 6,223 all-zero, 175 mixed and 2 all-correct groups, each with eight samples.
Exact `VERY_HARD`/`EASY` labels are not required for a diagnosis that establishes
excessive difficulty and the GRPO signal chain.
