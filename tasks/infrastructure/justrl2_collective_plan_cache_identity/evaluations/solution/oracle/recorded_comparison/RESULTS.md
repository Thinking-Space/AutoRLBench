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

| arm | wall / status | step 0 | step 9 | step 19 | step 29 | actor steps | `train/pg_loss` | `train/grad_norm` | `critic-value_loss` (0 → last) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| defective, seed 42 (shipped) | abort@step3 (`exit=1`) | 0.1035 | — | — | — | 0 | — | — | 0.122093 → 0.077107 |
| defective, seed 43 | 511 s, abort@step3 (`exit=1`) | 0.1045 | — | — | — | 0 | — | — | 0.107305 → 0.108007 |
| control (`fix.diff`), seed 42 | 629 s (`exit=0`) | 0.1016 | 0.2480 | 0.4131 | **0.4453** (`456/1024`) | 27 | `0.0000` | `1.45–2.38` | 0.098505 → 0.314224 |
| control (`fix.diff`), seed 43 | 634 s (`exit=0`) | 0.1006 | 0.2461 | 0.3242 | **0.4150** (`425/1024`) | 27 | `0.0000` | `1.51–2.57` | 0.117887 → 0.174614 |

Hardened `grade_submission.py` replay inside the digest-pinned container (`stage_integrity=true` with `pyarrow`) at `required_final_score = 0.32`:
- `defective_seed42`: `passed=false, score=0.0` (missing `val_generations/29.jsonl`, incomplete `0..3` rollout steps, `0` actor steps)
- `defective_seed43`: `passed=false, score=0.0` (missing `val_generations/29.jsonl`, incomplete `0..3` rollout steps, `0` actor steps)
- `control_seed42`: `passed=true, score=1.0, final_score=0.4453125` (`+0.1253` margin above `0.32`; `+0.3438` above control step 0)
- `control_seed43`: `passed=true, score=1.0, final_score=0.4150390625` (`+0.0950` margin above `0.32`; `+0.3145` above control step 0)

Cold end-to-end `8 × H100` verification (`CUDA_VISIBLE_DEVICES="" environment/prepare.sh` from SHA-verified HF cache -> `fix.diff` -> first-time GPU `convert_hf_to_torch_dist.py --attention-backend fused --no-gradient-accumulation-fusion` -> 30-step PPO -> `verifier/test.sh`):
- `control_seed42_e2e_cold` (job `1799775`): `passed=true, score=1.0, final_score=0.537109375` (`550/1024`), `stage_integrity=true`, `dump_problems=[]`.
