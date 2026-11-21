# Recorded comparison

All eight archived runs execute `run_grpo_countdown.sh` (45 steps, `test_freq=15`) on 1 × A800 (80 GB):

- `workspace/results/grpo_countdown` — shipped defective run (`build_optimizer` + `receive_weights`), `SEED=42`
- `defective_seed43/` — second defective run (`build_optimizer` + `receive_weights`), `SEED=43`
- `ablation_receiver_fixed_seed42/` — `receive_weights` fixed, `build_optimizer` still defective, `SEED=42`
- `ablation_receiver_fixed_seed43/` — `receive_weights` fixed, `build_optimizer` still defective, `SEED=43`
- `ablation_optimizer_fixed_seed42/` — `build_optimizer` fixed, `receive_weights` still defective, `SEED=42`
- `ablation_optimizer_fixed_seed43/` — `build_optimizer` fixed, `receive_weights` still defective, `SEED=43`
- `control_seed42/` — full repair (`fix.diff`), `SEED=42`
- `control_seed43/` — full repair (`fix.diff`), `SEED=43`

Held-out Countdown accuracy re-scored from `val_generations/<step>.jsonl` (`1,024` rows) with exact
rational arithmetic AST checking:

| arm | step 0 | step 15 | step 30 | step 45 | `actor/kl_loss` (step 1 → 45) | `actor/ppo_kl` (step 1 → 45) | `actor/entropy` (step 1 → 45) | `critic/score/mean` (step 1 → 45) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.0117 | 0.0146 | 0.0205 | 0.0410 | 0.00001 → 0.00207 | -0.00000 → -0.00001 | 0.688 → 0.673 | 0.0391 → 0.0371 |
| defective, seed 43 | 0.0117 | 0.0117 | 0.0156 | 0.0449 | 0.00001 → 0.00245 | 0.00001 → 0.00003 | 0.649 → 0.619 | 0.0195 → 0.0371 |
| fix `receive_weights` only, seed 42 | 0.0107 | 0.0137 | 0.0244 | 0.0459 | 0.00001 → 0.00464 | -0.00002 → 0.00004 | 0.653 → 0.598 | 0.0371 → 0.0527 |
| fix `receive_weights` only, seed 43 | 0.0117 | 0.0137 | 0.0186 | 0.0664 | 0.00001 → 0.00810 | -0.00002 → 0.00008 | 0.667 → 0.656 | 0.0313 → 0.0313 |
| fix `build_optimizer` only, seed 42 | 0.0107 | 0.1260 | 0.1592 | 0.1650 | 0.00030 → 0.73572 | -0.00001 → 0.01586 | 0.695 → 0.325 | 0.0234 → 0.1328 |
| fix `build_optimizer` only, seed 43 | 0.0117 | 0.1465 | 0.1553 | 0.2695 | 0.00044 → 1.92513 | 0.00020 → 0.07638 | 0.690 → 0.154 | 0.0313 → 0.1641 |
| control (`fix.diff`), seed 42 | 0.0117 | 0.2402 | 0.4072 | 0.4658 | 0.00032 → 0.06364 | 0.00008 → 0.00005 | 0.680 → 0.080 | 0.0352 → 0.4004 |
| control (`fix.diff`), seed 43 | 0.0107 | 0.2588 | 0.4063 | 0.4883 | 0.00031 → 0.08590 | -0.00001 → 0.00004 | 0.735 → 0.169 | 0.0371 → 0.3887 |
