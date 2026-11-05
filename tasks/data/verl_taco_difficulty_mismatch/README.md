# Training data is too hard

Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training parquet contains
only `VERY_HARD` problems, which the 1.5B model almost never solves under the
binary all-tests-pass reward, so nearly every GRPO group is all-zero, advantages
vanish on most steps, and held-out accuracy never improves.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | data |
| **Tags** | `verl` `grpo` `taco` `data-difficulty` `issue-only` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO: 2,329 `VERY_HARD` train / 184 `EASY` held-out |
| **Framework** | verl (GRPO), `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 4 GPUs |
| **Evaluations** | issue |
| **Recorded runs** | 1 defective run, 400/400 steps in 15 h 18 m (137.7 s/step); no control run — issue-only, no threshold to calibrate |

## Difficulty

The defect is in `workspace/taco_verl/train.parquet`: it holds 2,329 TACO
`VERY_HARD` rows (all `stdin`; 4 rows above 2048 prompt tokens dropped), while
`workspace/taco_verl/test.parquet` is the unchanged 184-row TACO `EASY`
validation set. The parquets do not store `extra_info.difficulty`, so difficulty
must be established from the problem statements and the pass-rate gap, not from
a label column. Code, launch hyperparameters (`lr=1e-6`, 16 prompts × 8 rollouts,
`use_kl_loss=True`, `kl_loss_coef=0.02` low_var_kl, `entropy_coeff=0.001`,
rollout temperature 1.0, validation temperature 0.7), and the model are the
healthy baseline.

Nothing fails loudly: the run completes all 400 steps (145 steps per epoch over
2,329 rows, so the step cap stops it in the third epoch) with valid shapes, and
entropy actually rises from 0.69 to about 1.0–1.1 (peak 2.04), which reads as
healthy exploration rather than a stall; on zero-advantage steps the only
remaining loss terms are the KL and entropy terms (`actor/kl_loss` grows slowly
to ~0.005), and held-out accuracy does not degrade. The single strongest piece
of evidence, readable from the bundled log and rollout dumps alone: with the
same model and the same binary all-tests-pass scorer, validation `EASY` rollouts
pass 22.5% (`val-aux/taco/pass_rate/mean@1` = 0.2246 at step 0) while training
rollouts pass 0.78% (104 of 2,151 distinct training prompts are ever solved) —
which pins the failure on training-data difficulty rather than the reward code
(`verl/utils/reward_score/taco.py`), the launch script, or the model.

The mechanism, from the frozen evidence: `critic/rewards/mean` averages 0.0078
over 400 steps (254 steps exactly 0, max 0.1016); only 175 of 6,400 GRPO groups
(2.7%) contain both a correct and an incorrect rollout, 2 are all-correct, the
rest all-zero; and 256 of 400 steps have all-zero advantages with
`actor/pg_loss` = 0 (`actor/grad_norm` 0.0037–0.020 on those steps). Held-out
accuracy on the 184 `EASY` rows stays flat within binomial noise (standard
error ≈ 0.019):

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `val-core/taco/acc/mean@1` | 0.0707 | 0.0924 | 0.0870 | 0.0870 | 0.0707 | 0.0815 | 0.0707 | 0.0652 | 0.0761 |

## Recorded run

- Log: `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261021_080557.log`
- Wandb: `workspace/logs/wandb/offline-run-20261021_080907-4pye3rsp`
- 4 GPUs, 400/400 steps in 15:18:01 (137.70 s/step); 145 steps per epoch (2329
  training rows), so the step cap stops training in the third epoch.
- Config: `train_batch_size=16`, `rollout.n=8`, `lr=1e-6`, rollout temperature
  1.0, validation temperature 0.7.
- `workspace/rollout_data/` holds the per-step training rollout dumps (400 files
  x 128 rollouts) and `workspace/validation_data/` the nine validation dumps
  (184 rollouts each) — the primary quantitative evidence for the causal chain.

## Reference solution

- **issue**: a passing diagnosis must name both parts: (1) the training data is
  too hard for this model — `train.parquet` is `VERY_HARD`-only against a 1.5B
  policy — and (2) the causal chain to the failure: the TACO reward is binary
  (1 only when every test passes), the policy almost never earns a 1, nearly
  every GRPO group of 8 is all-zero so its group-normalized advantage is exactly
  0, the policy-gradient term vanishes on most steps (256/400), and training
  never learns. The train/validation difficulty mismatch is how the agent can
  tell the training set is too hard; it is not the mechanism (the run would stall
  the same way with a `VERY_HARD` validation set). Stating only "too hard" or
  only the mismatch, without the causal chain, earns partial root-cause credit
  (0.3) and cannot pass. Attributions that must fail: KL/entropy settings,
  learning rate, batch shape, response length (300–750 tokens, clip ratio 0.2%),
  reward hacking (hard-coded `print(<constant>)` outputs in 9/51,200 training
  and 0/1,656 validation rollouts), evaluation artifact, and code defects in
  `taco.py`, the launch script, or `verl/utils/torch_functional.py`.

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum:
  `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
  Issue evaluation launches no training and needs no GPU. There is a single
  defective run and no control arm, so no threshold was calibrated and
  run-to-run variance was not measured.

## Packaging notes

These notes belong here rather than in `environment/prepare.sh`, which is
mounted read-only into the agent stage.

**Assets.** Dataset: `LIF1014/autorlbench-data-task-train-too-hard-dataset` @
`22a1fee0ea2527f514c4115f6c792216a99d7e6f` (Apache-2.0), archive
`artifacts/workspace-assets.tar.gz` of 167,612,568 bytes, SHA-256
`eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916`. Model:
`Qwen/Qwen2.5-1.5B-Instruct` @
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (`model.safetensors` SHA-256
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`),
byte-identical on every allowed file to the former task snapshot
`LIF1014/autorlbench-verl-implementation-torch-functional@36a4c22`; the source
was switched because that repo name is visible to the agent through the shared
download cache.

| Runtime path | Size | SHA-256 |
| --- | --- | --- |
| `workspace/taco_verl/test.parquet` | 61,082,426 bytes | `0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3` |
| `workspace/taco_verl/train.parquet` | 168,834,942 bytes | `8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573` |

`environment/prepare.sh` fetches the whole repos via the task-local
`fetch_hf.py` (cache-first, no checksum verification); the pinned revisions,
sizes, and SHA-256 values above are kept for reference.

**Known leak.** The dataset repository name above contains the answer and
appears in the agent-visible `environment/prepare.sh`. Republishing the archive
under a neutral repository name is still open.

**`pyext`.** The recorded run had `pyext==0.7` installed (see the wandb
requirements). Issue evaluation does not run the scorer, so `prepare.sh` does
not install it.

**Removed run.** An earlier version of this task shipped a run with
`use_kl_loss=False` and `entropy_coeff=0`; it failed differently (reward hacking
with hard-coded outputs, entropy and length collapse, validation below step 0),
which made "no KL / no entropy regularization" an independently sufficient
explanation. It has been removed from the workspace; only the run above is
shipped.

Audit note (2026-10-31): all 51,200 training records parse when read as
LF-delimited JSONL. One response in `workspace/rollout_data/380.jsonl` contains
a legal Unicode U+2028 character; `str.splitlines()` incorrectly splits that
record. Group by prompt, not adjacent blocks of eight: the intact evidence
has 6,223 all-zero, 175 mixed and 2 all-correct groups, each with eight samples.
Exact `VERY_HARD`/`EASY` labels are not required for a diagnosis that establishes
excessive difficulty and the GRPO signal chain.
