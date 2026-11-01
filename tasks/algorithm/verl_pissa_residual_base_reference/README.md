# PiSSA residual base used as the KL reference

This solution task stages a real four-update GRPO experiment on
Qwen2.5-0.5B-Instruct with a rank-8 PiSSA adapter. The defective code treats
the residual checkpoint left by PiSSA initialization as the pretrained base
when computing the KL reference.

| | |
| --- | --- |
| **Author** | Ruijia Zhang |
| **Domain** | algorithm |
| **Framework** | verl, FSDP actor + asynchronous vLLM rollout |
| **Data** | fixed GSM8K subset: 16 train / 8 validation |
| **Budget** | 4 GRPO updates, rollout n=4, response length 32, 1 GPU |
| **Environment** | `verl-vllm011` (not yet built) |
| **Recorded runs** | 1 defective + 1 repaired run |

## Defect and evidence

PiSSA rewrites each adapted layer as `W_residual = W_pretrained - delta_init`.
Consequently, `disable_adapter()` does not recover `W_pretrained`. In the
recorded defective run, `val-health/base_reference_consistency` is
`0.045778327704709` at step 4. The otherwise identical repaired run evaluates
the frozen initial PiSSA adapter and reaches exactly `1.0`.

The workspace includes the full source snapshot, the exact launch script,
metrics for steps 0–4, offline W&B logs, first/middle/final rollout and
validation samples, debug probes, environment freeze and GPU metadata.

## Verification

The solution harness runs `bash run_experiment.sh` and grades only fresh files
under `$AUTORLBENCH_RESULTS/experiment_logs`. Step 4 must report
`val-health/base_reference_consistency >= 0.95`. The threshold is calibrated
from one run per side: margin `0.9042216723` above the defective run and `0.05`
below the repaired run. Step-zero health was not logged; the exact step-4
requirement prevents an unexecuted run from passing, but cross-seed variance is
not established.
