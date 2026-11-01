# Improve the staged experiment

Review the staged experiment and determine whether it is healthy. Inspect the
source, launch script, metrics, rollouts and debug evidence. If you find a
defect, repair it in `/workspace`. Do not launch training; the system performs
the graded rerun after your response. Cheap non-training checks are allowed.

Write `/workspace/report.md` describing the root cause, every changed file and
the expected metric effect. A missing report scores zero.

The system deletes `$AUTORLBENCH_RESULTS/experiment_logs`, then runs:

```bash
bash run_experiment.sh
```

The fixed experiment must preserve the supplied Qwen2.5-0.5B PiSSA rank-8
bundle, deterministic GSM8K 16/8 subset, rollout n=4, response length 32,
temperature 0.8, KL coefficient 0.05, four GRPO updates, seed and one-GPU
layout. Do not add format reward or any auxiliary reward, and do not alter
data, model, budget or metric logging. The grader reads only fresh files under
`experiment_logs` and requires the final step-4 actor/rollout health metric;
copying or fabricating historical metrics scores zero.
