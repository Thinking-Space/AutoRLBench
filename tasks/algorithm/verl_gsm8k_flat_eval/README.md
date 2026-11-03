# Flat Held-Out GSM8K Score in VERL GRPO

Training Qwen3-1.7B with GRPO on GSM8K can produce a flat held-out score across a 50-step run.

| | |
| --- | --- |
| **Author** | Xuyang Yuan |
| **Domain** | algorithm |
| **Model** | Qwen/Qwen3-1.7B |
| **Dataset** | OpenAI GSM8K main: 7,473 train / 1,319 test; first 256 held-out rows |
| **Framework** | verl, `run.sh` |
| **Environment** | `verl` · 1 GPU |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 observed + 1 baseline run, 50 steps each; a separate baseline step-0 evaluation |

## Difficulty

The defect is a one-condition implementation detail in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: a generic config-to-`SamplingParams` loop accepts the `seed` field because that field exists in the installed vLLM API. The code remains runnable and the failure is behavioral rather than an exception.

The observed run's actor gradient norm and policy-gradient loss are zero at all 50 steps, and all ten held-out evaluations report `0.734375`. The baseline workspace scores `0.73828125` at step 0 and varies from `0.75390625` to `0.8828125` over training. One run per arm is evidence for this snapshot, not a statistical estimate.

## Reference solution

- **issue**: diagnose the rollout-seed forwarding in the shared `SamplingParams` config loop and connect it to repeated sampling / loss of useful GRPO group signal. The flat metric is evidence, not the source-level cause.
- **solution**: exclude the `seed` key from the generic `SamplingParams` copy, then complete the same 50-step run.

## Verification

- **issue**: an LLM judge compares the diagnosis with the hidden oracle on verdict / root cause / evidence. Passing requires `verdict_score == 1.0` plus `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent repairs the code and submits `/workspace/report.md` without launching training (`run_spec.json`, debug-then-report). The system deletes `$AUTORLBENCH_RESULTS/experiment_logs`, re-runs `/workspace/run.sh` in the repaired workspace, and the final `val-core/openai/gsm8k/reward/mean@1` at step 50 of that fresh run must be at least `0.74609375`. It is `0.01171875` above the observed run's maximum, `0.0078125` below the baseline run's minimum, and `0.0078125` above baseline step 0. Threshold margins are calibrated on one run per arm; variance was not measured.

The metric gate alone does not distinguish a root-cause repair from a masking fix (dropping the `rollout.seed` override from the launch script also clears it). The solution instruction forbids changing the recorded launch configuration, and the issue-level diagnosis is where the root cause is graded.
