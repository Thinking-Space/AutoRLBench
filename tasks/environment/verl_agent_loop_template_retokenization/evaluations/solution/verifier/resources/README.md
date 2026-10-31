`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs
`training/rollout_actor_probs_pearson_corr` from `$AUTORLBENCH_RESULTS/experiment_logs`.

- `metric_key`: `training/rollout_actor_probs_pearson_corr`
- `score_mode`: `final`, the value at `final_step`
- `final_step`: `100`, the last step of the fixed 100-step budget
- `required_final_score`: `0.9`
- `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.0071`, the defective run's value at step 100

Pass is score `1`. All failures are score `0`.
