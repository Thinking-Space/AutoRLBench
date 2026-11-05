# Improve the staged experiment

Review the staged experiment and determine whether it is healthy. Inspect the
source, launch script, metrics, rollouts and debug evidence. If you find a
defect, repair it in `/workspace`. Do not launch training; the system performs
the graded rerun after your response. Cheap non-training checks are allowed.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.

Write `/workspace/report.md` describing the root cause, every changed file and
the expected metric effect. A missing report scores zero.

The system deletes `$AUTORLBENCH_RESULTS/experiment_logs`, then runs:

```bash
bash run_experiment.sh
```

The fixed experiment must preserve the supplied Qwen2.5-0.5B PiSSA rank-8
bundle, deterministic GSM8K 16/8 subset, rollout n=4, response length 32,
temperature 0.8, KL coefficient 0.05, four GRPO updates, seed and one-GPU
layout. Do not alter rewards, data, model, budget or metric logging. The grader
reads only fresh files under `experiment_logs`: the final-step
validation health metrics, cross-checked against verl's own per-step file
log (including the per-update KL loss) and the rollout sync probe the run
writes, with the launch script and the health instrumentation pinned by
sha256. Copying or fabricating metrics scores zero.
