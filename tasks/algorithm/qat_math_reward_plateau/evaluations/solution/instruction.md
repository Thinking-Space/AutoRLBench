# Diagnose and repair the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Inspect the effective configuration, launch scripts, reward
logic, training source, and recorded rollouts and metrics. Use held-out
evaluation evidence together with the training signals.

If you find a defect, fix it in the staged workspace at `/workspace`.
Keep the initial model, training data order, seed, sampling, reward scale,
and evaluation protocol. Do not launch training or use the network. Cheap,
non-training checks are fine.

Submit a short report in `/workspace/report.md` with the verdict, evidence
for each root cause, changes, and expected effect on training.

The harness runs `bash train.sh` from the initial model through step 30 with
64 prompts and one 16,384-token answer budget per prompt, on four GPUs.
It then evaluates the step-30 checkpoint using `eval_aime24.sh` on the
fixed 30 AIME24 questions with eight draws per question. The verifier uses
the fresh checkpoint, the last ten steps of training metrics, and the
held-out score.
