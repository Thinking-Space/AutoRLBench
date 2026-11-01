# PiSSA adapter omitted from vLLM rollout requests

This solution task stages a real four-update GRPO experiment on
Qwen2.5-0.5B-Instruct with a rank-8 PiSSA adapter. The defective vLLM request
path fails to attach the adapter even though it is loaded in the engine.

| | |
| --- | --- |
| **Author** | Ruijia Zhang |
| **Domain** | infrastructure |
| **Framework** | verl, FSDP actor + asynchronous vLLM rollout |
| **Data** | fixed GSM8K subset: 16 train / 8 validation |
| **Budget** | 4 GRPO updates, rollout n=4, response length 32, 1 GPU |
| **Environment** | `verl-vllm011` (not yet built) |
| **Recorded runs** | 1 defective + 1 repaired run |

## Defect and evidence

`engine.list_loras()` returns integer IDs. Comparing that set with the string
adapter name makes `lora_loaded` false, so vLLM samples from the residual base
while the actor scores the PiSSA policy. The defective run's step-4
`val-health/actor_rollout_policy_agreement` is `0.10476375009925555`; the
one-line integer-ID repair reaches `0.8907221660212565`.

The defective arm has zero GSM8K reward and gradient because its residual-base
rollouts are poor. This is an observed effect, not the graded root cause and
not a training crash: all four rollout/update/validation steps completed. No
format or auxiliary reward was added, so the control variable remains the
adapter request.

## Verification

The solution harness runs `bash run_experiment.sh` and grades only fresh files
under `$AUTORLBENCH_RESULTS/experiment_logs`. Step 4 must report
`val-health/actor_rollout_policy_agreement >= 0.80`. The threshold has margins
`0.6952362499` above the defective run and `0.0907221660` below the repaired
run. Calibration uses one run per side; step-zero health and cross-seed
variance are unavailable.
