# PiSSA residual base used as the KL reference

This solution task stages a real four-update GRPO experiment on
Qwen2.5-0.5B-Instruct with a rank-8 PiSSA adapter. The defective code treats
the residual checkpoint left by PiSSA initialization as the pretrained base
when computing the KL reference.

| | |
| --- | --- |
| **Author** | Ruijia Zhang |
| **Affiliation** | Johns Hopkins University |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `pissa` `kl-reference` `residual-base` `masked-repair` |
| **Framework** | verl (GRPO), FSDP actor + asynchronous vLLM rollout |
| **Run time** | ≤ 1 hour |
| **Resources** | 1 GPU |

## Difficulty

PiSSA rewrites each adapted layer as `W_residual = W_pretrained - delta_init`.
Consequently, `disable_adapter()` does not recover `W_pretrained`. The
otherwise identical repaired run evaluates the frozen initial PiSSA adapter
and reaches `val-health/base_reference_consistency` exactly `1.0`.

The frozen `observed_run` predates the validation-health probe: the recorded
log, `metrics/all_steps.jsonl` and `artifact_manifest.json` do not print any
`val-health/*` values (the graded rerun, running the pinned current code,
does). The natural symptom is independent of that metric: the KL reference
equals the initial policy, so a correct run logs `actor/kl_loss` near zero at
step 1, while the defective run logs `0.7618` and stays above `0.76` for all
four updates (`actor/base_model_kl` around `-4.6` similarly). Both signals are
visible in `metrics/all_steps.jsonl` and the text log.

The workspace includes the full source snapshot, the exact launch script,
metrics for steps 0–4, offline W&B logs, first/middle/final rollout and
validation samples, debug probes, environment freeze and GPU metadata.

The recorded evidence also supports plausible-but-wrong diagnoses, which the
issue oracle lists: the low validation accuracy (a 0.5B model on 8 problems),
the short four-update run, the KL coefficient, the offline W&B shutdown
warnings at the end of `train.log`, or a generic claim that training failed.
Actor/vLLM recomputation agreement (`rollout_actor_probs_pearson_corr` `0.99`)
rules out rollout synchronization. None of
these explains the step-1 `actor/kl_loss` of `0.7618`.

No blind runs have been recorded; difficulty is not yet characterized
empirically.

## Reference solution

- **issue**: the agent must connect three things visible in the frozen
  evidence — the `lora_residual_base=True` launch flag with the
  `residual_base`/`pissa_adapter` bundle paths, the step-1 `actor/kl_loss` of
  `0.7618` (the reference should equal the initial policy), and the
  `_compute_base_log_prob_from_actor` code path in the vendored worker — and
  must know (or derive from the PiSSA setup) that `disable_adapter()` on a
  PiSSA residual base exposes `W_residual`.
- **solution**: the repair must rebuild the base reference from the frozen
  initial adapter (or an equivalent reconstruction); disabling the adapter
  everywhere or flipping the residual-base flag is a masked repair and fails
  the KL corroboration.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle,
  scoring `verdict_score`, `root_cause_score` and `evidence_score` from 0 to
  1. Passing requires `verdict_score == 1.0` with both `root_cause_score` and
  `evidence_score` above `0.8`; the 0.2 / 0.6 / 0.2 weighted sum is reported
  as the score but is not the pass rule.
- **solution**: the solution harness runs `bash run_experiment.sh` and grades
  only fresh files under `$AUTORLBENCH_RESULTS/experiment_logs`. Step 4 must
  report `val-health/base_reference_consistency >= 0.95`, anchored four ways
  beyond the metric bar: (1) verl's per-step file log must show a complete run
  whose final `mae`/`consistency` pair satisfies `consistency == exp(-mae)` and
  matches the judged text-log value; (2) every training step must keep
  `actor/kl_loss` under 0.1 (recorded repaired `0.0` vs defective
  `0.7618-0.9751`), which rejects masked repairs such as re-running with
  `lora_residual_base=False` — that compares the wrong reference against
  itself and vacuously reaches consistency `1.0` while leaving the KL target
  wrong; (3) the rollout sync probe must still show the PiSSA actor
  (`is_svd_init`, `lora_delta`, the PiSSA adapter path); (4) the launch script
  and the health instrumentation are sha256-pinned at grading time, so budget,
  flag or metric-computation edits fail. A single fabricated log line no
  longer passes.

  The threshold is calibrated from one run per side: margin `0.9042216723`
  above the defective run and `0.05` below the repaired run. Step-zero health
  was not logged; the exact step-4 requirement prevents an unexecuted run from
  passing, but cross-seed variance is not established.
