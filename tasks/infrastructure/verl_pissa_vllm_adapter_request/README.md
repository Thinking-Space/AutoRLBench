# PiSSA adapter omitted from vLLM rollout requests

This solution task stages a real four-update GRPO experiment on
Qwen2.5-0.5B-Instruct with a rank-8 PiSSA adapter. The defective vLLM request
path fails to attach the adapter even though it is loaded in the engine.

| | |
| --- | --- |
| **Author** | Ruijia Zhang |
| **Affiliation** | Johns Hopkins University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `type-mismatch` `adapter-omission` `actor-rollout-divergence` |
| **Framework** | verl (GRPO), FSDP actor + asynchronous vLLM rollout |
| **Run time** | ~2.5 minutes |
| **Resources** | 1 GPU |

## Difficulty

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

## Reference solution

- **issue**: identify the string-name versus integer-ID mismatch that
  suppresses the vLLM `LoRARequest` — `engine.list_loras()` returns integer
  IDs, the request path compares that set with the string adapter name, so
  `lora_loaded` is false and rollout requests omit the loaded PiSSA adapter
  while the actor evaluates with it.
- **solution**: the one-line integer-ID repair, so the adapter is attached to
  rollout requests; the step-4 `val-health/actor_rollout_policy_agreement`
  rises from `0.10476375009925555` to `0.8907221660212565`.

## Verification

- **issue**: the LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden oracle. Verdict, root cause and evidence are weighted
  0.2 / 0.6 / 0.2; passing requires at least 0.8. The essential attribution is
  the string-name versus integer-ID mismatch that suppresses the vLLM
  `LoRARequest`.
- **solution**: the solution harness runs `bash run_experiment.sh` and grades
  only fresh files under `$AUTORLBENCH_RESULTS/experiment_logs`. Step 4 must
  report `val-health/actor_rollout_policy_agreement >= 0.80`. The threshold has
  margins `0.6952362499` above the defective run and `0.0907221660` below the
  repaired run. Calibration uses one run per side; step-zero health and
  cross-seed variance are unavailable.

Because the metric is computed by instrumentation inside the shipped
workspace, the grader anchors the value: verl's per-step file log must show a
complete run whose final row satisfies `agreement == exp(-mae)` and matches
the judged text-log value; the rollout sync probe must still show the PiSSA
actor; a majority of the per-request generate probes must show the adapter
attached (rejecting the masked repair of disabling it on both sides); and live
grading sha256-pins `run_experiment.sh` and the health instrumentation in
`ray_trainer.py`. Not defended: a post-hoc forgery that fabricates the whole
self-consistent record. The image is not yet built; a GPU qualification run
must confirm the grader accepts a real repaired run end-to-end.
