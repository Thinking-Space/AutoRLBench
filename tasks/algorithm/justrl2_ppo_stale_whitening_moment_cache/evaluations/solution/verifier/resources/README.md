# Solution verifier resources

Entry point is `../test.sh`, which runs `../tests/grade_submission.py`. The grader reads only fresh rerun outputs under `$AUTORLBENCH_RESULTS/experiment_logs` (ignoring frozen workspace logs) against `grader_spec.json`:

- `metric_key`: `heldout/countdown_exact_acc@29` (`val_generations/29.jsonl`, Countdown exact AST + rational `Fraction` check, `>= 1,000` rows)
- `final_step`: `29` (`expected_train_steps = 30`, `num_critic_only_steps = 3`, `eval_interval = 10`)
- `required_final_score`: `"0.28"` (`historical_final_baseline = "0.2129"`, `required_final_score_max = "1.0"`)
- `runner_pinned`: `true` (`run_justrl2_ppo.sh` byte-matched against `resources/run_justrl2_ppo.sh`), plus SHA-256 and raw Countdown slice checks on `data_prep/countdown.py`, `reward/countdown_reward.py`, `raw/countdown/data/train-00000-of-00001.parquet`, and `data/countdown/{train,test}.jsonl`.
