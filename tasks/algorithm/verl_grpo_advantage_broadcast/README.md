# algorithm diagnosis: TACO GRPO run

Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The per-sample advantage vector is laid along the sequence axis and broadcast across the batch before the response mask, so the policy gradient is decoupled from each sample's reward and held-out accuracy never improves.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Type** | algorithm |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO EASY (8368 train / 184 test) |
| **Framework** | verl (GRPO), `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 8 GPUs |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective run bundled (284 of 400 steps, ~8 h, job time limit); protocol shortened to 300 steps and thresholds recalibrated 2026-11-03 from a fresh same-revision A/B — repaired arm (devspace 1446084) vs two defective arms (devspace 1177503 full 400 steps; devspace 1236983, 241 steps, preempted) |

## Difficulty

The defect is in `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`. After the per-uid mean/std normalization, the length-`B` advantage vector is reshaped to `[1, B]`, padded or truncated to the response length, and expanded across the batch:

```python
seq_len = response_mask.size(1)
row = scores.reshape(1, -1)
if row.size(1) < seq_len:
    row = torch.nn.functional.pad(row, (0, seq_len - row.size(1)))
else:
    row = row[:, :seq_len]
scores = row.expand(response_mask.size(0), -1) * response_mask
```

With train batch 16 × rollout `n=8`, `B=128` while `max_response_length=4096`: column `j` carries sample `j`'s advantage on every row, and positions past 128 get 0 before the mask. The returned shape is still `[B, S]`, so nothing raises; the defect is advantage values, not a gradient leak (the layout sits inside the existing `torch.no_grad()` block).

Nothing fails loudly: shapes match the policy loss, the run is stable (`actor/entropy` 0.39–0.81, mean response length 149–503 tokens, clip ratio ≈ 0), and training reward is present — `critic/rewards/mean` averages 0.122 over steps 1–284 with no zero step, so "no learning signal" is a wrong diagnosis. The single strongest piece of evidence, readable from the bundled log alone: the token-weighted `critic/advantages/mean` stays near zero on every step (median absolute value 4.9e-4) while rewards are nonzero — correctly aligned advantages track the length/reward correlation at 3.9e-3–9.3e-3 on the correct-code TACO runs (`task_train_too_hard_new`, `task_function_call`, `torch_functional`). Each row sums a prefix of the 128 group-normalized values, which sum to ≈ 0, instead of its own advantage times its length.

| step | 0 | 50 | 100 | 150 | 200 | 250 |
| --- | --- | --- | --- | --- | --- | --- |
| `val-core/taco/acc/mean@1` (defective) | 0.0761 | 0.0489 | 0.0598 | 0.0707 | 0.0435 | 0.0543 |

Over the same run, `critic/rewards/mean` averages 0.122 (steps 1–284, fifty-step means 0.093–0.133), while the token-weighted `critic/advantages/mean` never leaves ±4.9e-4 of zero.

## Reference solution

- **issue**: a passing diagnosis names `compute_grpo_outcome_advantage` and the layout defect — the length-`B` advantages reshaped to `[1, B]`, padded to the response length, and expanded across rows, so column `j` carries sample `j`'s advantage for every row. Attributions that must fail: the run stopping at step 284 of 400 (truncated job, no exception), `use_kl_loss=False` / `entropy_coeff=0` (shared launch defaults for this TACO family; entropy and length do not collapse), scorer runtime errors in `[error] runtime: ...` lines (exceptions inside generated programs, caught and recorded as failures), learning rate, batch size, the dataset, and a `keepdim`/`unsqueeze(0)` form — that form would raise under `B=128`, `S=4096`.
- **solution**: restore `scores = scores.unsqueeze(-1) * response_mask` after the per-uid normalization, so row `i` carries sample `i`'s scalar advantage on its response tokens. This is sufficient because it is the only change between the defective workspace and the healthy baseline file.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the rerun (300 steps, `TEST_FREQ=50`) is scored on two components — validation `val-core/taco/acc/mean@1` **mean over the pinned steps 200, 250, 300** in **[0.095, 0.35]** (weight 0.8) and mean `critic/rewards/mean` over training steps **1..300** in **(0.155, 0.6]** (weight 0.2); all pinned steps and all 300 training steps must be present, and pass requires `score == 1.0` (both components). Calibration (same revision, same launch, one run per arm): repaired run means **0.1105 / 0.1763**; the full 400-step defective rerun means **0.0815 / 0.1397** over the same window, and a second defective rerun (241 steps, preempted) agrees. The mean over pinned steps replaces the earlier best-peak rule: on the 184-row validation set a single point carries SE ≈ 0.019, and the fresh defective run's peak hit 0.0978 (18/184) — one correct answer from the old 0.10 peak bar — while its pinned mean stays 0.0815; pinning the steps also closes the `TEST_FREQ` / `TOTAL_TRAINING_STEPS` short-run shortcut, since both are plain env overrides in the launch script. The ceilings (0.35 / 0.6) guard against a scorer rewritten to a constant reward, since both graded quantities come from agent-repairable code. Margins only, no variance — one run per arm.
