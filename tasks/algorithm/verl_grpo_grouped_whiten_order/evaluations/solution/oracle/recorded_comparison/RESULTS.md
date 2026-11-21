# Recorded comparison

All eight archived runs execute `run_grpo_gsm8k.sh` (50 steps, `test_freq=25`) on 1 × A800 (80 GB):

- `workspace/results/grpo_gsm8k` — shipped defective run (`grouped_whiten` + `kl_penalty`), `SEED=42`
- `defective_seed43/` — second defective run (`grouped_whiten` + `kl_penalty`), `SEED=43`
- `ablation_whiten_only_seed42/` — `kl_penalty` fixed, `grouped_whiten` still defective, `SEED=42`
- `ablation_whiten_only_seed43/` — `kl_penalty` fixed, `grouped_whiten` still defective, `SEED=43`
- `ablation_seq_kl_only_seed42/` — `grouped_whiten` fixed, `kl_penalty` STE still defective, `SEED=42`
- `ablation_seq_kl_only_seed43/` — `grouped_whiten` fixed, `kl_penalty` STE still defective, `SEED=43`
- `control_seed42/` — full repair (`fix.diff`), `SEED=42`
- `control_seed43/` — full repair (`fix.diff`), `SEED=43`

Held-out GSM8K accuracy re-scored from `val_generations/<step>.jsonl` (`1,319` rows) with the
`last_number` rule:

| arm | step 0 | step 25 | step 50 | `actor/kl_loss` (step 1 → 50) | `actor/entropy` (step 1 → 50) | `critic/score/mean` (step 1 → 50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4625 | 0.4579 | 0.4610 | 0.00029 → 0.00795 | 0.485 → 0.456 | 0.4102 → 0.3594 |
| defective, seed 43 | 0.4625 | 0.4617 | 0.4640 | 0.00025 → 0.00635 | 0.478 → 0.419 | 0.3867 → 0.3691 |
| fix `kl_penalty` only (`grouped_whiten` active), seed 42 | 0.4625 | 0.4481 | 0.4102 | 0.00027 → 0.05804 | 0.497 → 0.398 | 0.4160 → 0.3164 |
| fix `kl_penalty` only (`grouped_whiten` active), seed 43 | 0.4632 | 0.4261 | 0.3654 | 0.00028 → 0.07011 | 0.497 → 0.220 | 0.3418 → 0.4004 |
| fix `grouped_whiten` only (`kl_penalty` STE active), seed 42 | 0.4617 | 0.4829 | 0.4807 | 0.00028 → 0.00748 | 0.526 → 0.393 | 0.3906 → 0.4180 |
| fix `grouped_whiten` only (`kl_penalty` STE active), seed 43 | 0.4647 | 0.4754 | 0.4905 | 0.00030 → 0.00756 | 0.476 → 0.355 | 0.3535 → 0.4473 |
| control (`fix.diff`), seed 42 | 0.4587 | 0.5277 | 0.5588 | 0.00028 → 0.05953 | 0.484 → 0.103 | 0.4141 → 0.6875 |
| control (`fix.diff`), seed 43 | 0.4663 | 0.5254 | 0.5368 | 0.00030 → 0.08392 | 0.474 → 0.084 | 0.3809 → 0.6934 |
