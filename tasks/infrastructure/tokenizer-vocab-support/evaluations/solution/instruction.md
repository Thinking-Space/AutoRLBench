# Diagnose and repair the staged GRPO run

Inspect the source, configuration, and frozen training evidence in `/workspace`. Decide whether the run is healthy. Connect the mechanism you identify to observable behavior, including the difference between training and held-out metrics. Repair the defect in the staged workspace.

Do not launch training or use the network. Small offline checks are permitted. Preserve the model, dataset, reward and scoring code, evaluation settings, checkpoint compatibility, and launch protocol. Do not edit, copy, or fabricate metrics or evaluation responses.

Write `/workspace/report.md` with your verdict and supporting evidence, the root cause, each file changed, and the expected effect of your repair.

## Harness rerun

The harness invokes `bash scripts/train.sh --resume "$AUTORLBENCH_PREPARED/global_step_725"` with the paths and overrides in `run_spec.json`. It restores the native model, optimizer, scheduler, RNG, and data cursor on two GPUs. It executes 275 updates, from step 726 through step 1000, preserving the 1,000-step schedule, four optimizer minibatches per step, 64 train prompts with eight draws per prompt, and the 16,384-token response budget. It saves every 30 steps and at the native terminal step. It evaluates at step 1000 on AIME24, AIME25, and AMC23 with 32 draws per question.

The rerun timeout is 126,000 seconds. The harness writes fresh logs, validation data, and checkpoints below `$AUTORLBENCH_RESULTS/experiment_logs`. The verifier checks protected experiment inputs, independently scores the step-1000 responses, and checks their agreement with `val-core/overall/acc/mean@32`. A missing final evaluation fails the task.
