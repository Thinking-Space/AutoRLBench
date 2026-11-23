# Recorded comparison

All four calibration runs execute `bash run_justrl2_ppo.sh` (`STEPS=30`, `EVAL_INTERVAL=10`,
`NUM_CRITIC_ONLY_STEPS=3`, `ROLLOUT_BATCH_SIZE=16`, `N_SAMPLES_PER_PROMPT=4`, `MICRO_BATCH_SIZE=4`,
`CP=2 × DP=2`, `actor=4, critic=4`) on single-node `8 × H100 (80 GB)` inside
`ghcr.io/merak0514/autorlbench-justrl2@sha256:8d2eb1fe292a4e381915734b7854213002ca9bae6b50efab1bd8cc75cad14a8b`:

- `defective_seed42/` (also archived in `workspace/results/ppo_countdown/`) — shipped defective codebase, `SEED=42`
- `defective_seed43/` — shipped defective codebase, `SEED=43`
- `control_seed42/` — repaired with `evaluations/solution/oracle/fix.diff`, `SEED=42`
- `control_seed43/` — repaired with `evaluations/solution/oracle/fix.diff`, `SEED=43`

Held-out Countdown accuracy comes from `grade_submission.py` re-scoring `val_generations/<step>.jsonl` (`1,024` rows)
matching `eval/countdown` in `metrics.jsonl`:

| arm | wall / status | step 0 | step 9 | step 19 | step 29 | `trunc@29` (`len@29`) | `train/pg_loss` (min .. last) | `train/grad_norm` (max / last) | `ess` / `ppo_kl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| defective, seed 42 (shipped) | 639 s (`exit=0`) | 0.1006 | 0.2910 | 0.2666 | **0.2129** (`218/1024`) | 0.7275 (`416.9`) | `-3070.99` .. `-345.71` | `9669.23` / `2525.50` | `1.0` / `0.0` |
| defective, seed 43 | 642 s (`exit=0`) | 0.0977 | 0.2324 | 0.1309 | **0.0342** (`35/1024`) | 0.7656 (`426.1`) | `-679.17` .. `-599.91` | `3440.76` / `1490.59` | `1.0` / `0.0` |
| control (`fix.diff`), seed 42 | 628 s (`exit=0`) | 0.0996 | 0.0566 | 0.3564 | **0.4170** (`427/1024`) | 0.5205 (`318.1`) | `0.0000` .. `-0.0000` | `4.42` / `1.49` | `1.0` / `0.0` |
| control (`fix.diff`), seed 43 | 636 s (`exit=0`) | 0.0977 | 0.2188 | 0.2822 | **0.3447** (`353/1024`) | 0.6152 (`402.4`) | `-0.0000` .. `-0.0000` | `2.98` / `1.34` | `1.0` / `0.0` |

Selected seed-42 defective trajectory in `metrics.jsonl`:
- Step 3 (first actor update, fresh cache on non-empty/zero start): `train/pg_loss = 0.0`, `train/grad_norm = 2.82`.
- Step 4: `train/pg_loss = -623.6117`, `train/grad_norm = 3523.99`.
- Step 14–15: `train/grad_norm` peaks at `9669.23` (step 14); step 15 `pg_loss = -2947.4529`, `grad_norm = 8364.32`.
- Step 24–29: `train/pg_loss` reaches `-3070.9916` (step 24) and ends at `-345.7063` (step 29), while `train/critic-value_loss` stays `0.0187–0.1417`, `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, and `train/train_rollout_kl = 0.001113`.

Hardened `grade_submission.py` replay inside the digest-pinned container (`stage_integrity=true` with `pyarrow`) at `required_final_score = 0.28`:
- `defective_seed42`: `passed=false, score=0.0, final_score=0.212890625` (`-0.0671` below `0.28`)
- `defective_seed43`: `passed=false, score=0.0, final_score=0.0341796875` (`-0.2458` below `0.28`)
- `control_seed42`: `passed=true, score=1.0, final_score=0.4169921875` (`+0.1370` margin above `0.28`; `+0.3174` above control step 0)
- `control_seed43`: `passed=true, score=1.0, final_score=0.3447265625` (`+0.0647` margin above `0.28`; `+0.2471` above control step 0)

Cold end-to-end `8 × H100` verification (`CUDA_VISIBLE_DEVICES="" environment/prepare.sh` from SHA-verified HF cache -> `fix.diff` -> first-time GPU `convert_hf_to_torch_dist.py --attention-backend fused --no-gradient-accumulation-fusion` -> 30-step PPO -> `verifier/test.sh`):
- `control_seed42_e2e_cold` (job `1799778`): `passed=true, score=1.0, final_score=0.416015625` (`426/1024`), `stage_integrity=true`, `dump_problems=[]`.
