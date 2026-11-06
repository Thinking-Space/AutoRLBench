# Canonical solution oracle

## Root cause

- `run_gsm8k_grpo.sh` (`ADAM_EPS=0.001` → `actor_rollout_ref.actor.optim.override_optimizer_config={eps:0.001}`)
  - kind: `config`
  - observed: the override reaches `torch.optim.AdamW`. With per-parameter RMS gradients of about
    1e-5 to 5e-5, eps dominates `sqrt(v_hat) + eps`, so the update is `(lr / eps) * m_hat`, 10 to 100
    times smaller than lr 2e-6 gives with the default eps. The run learns the format about 30 steps
    late and held-out accuracy ends at 0.486.
  - expected: AdamW's default eps.

## Required remediation

- Remove the `ADAM_EPS` line and the `override_optimizer_config` entry (`fix.diff`, `code_target/`),
  so AdamW uses eps 1e-8. Any eps well below the per-parameter gradient scale is equivalent. Keeping
  eps 1e-3 and raising the learning rate to compensate is a different optimizer and was not
  recorded; the grader only scores the rerun.

Machine grading reads only the new experiment logs. The grader re-scores the fresh run's
`experiment_logs/val_generations/100.jsonl` (1,319 greedy GSM8K test responses, at least 1,300 rows)
with a fixed rule, last `#### <number>` numerically equal to the gold answer, and requires accuracy
≥ 0.53 (≤ 1.0). Pass is score 1, anything else 0. The runner is where the repair goes, so it is not
byte-compared (`runner_pinned: false`); the GSM8K parquets and raw files are pinned by sha256 and
`metrics.jsonl` is required as run-reality evidence (one row per training step 0-100, the step-100
evaluation logged; values never scored).

## Recorded comparison (`recorded_comparison/`)

One run per side, seed 42, 1 × RTX PRO 6000 each, `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image (verlai/verl:vllm017.latest, same digest) through Apptainer.
`control/run100/` is the control run: the same workspace with `fix.diff` applied (recorded with the
pre-baked launcher and the same Ray limits on the command line, so the effective command is
identical). `RESULTS.md` has the side-by-side numbers.

| step-100 held-out | strict accuracy |
| --- | ---: |
| defective | 0.493 (verl's logged strict metric 0.486) |
| control | 0.566 |
| threshold | ≥ 0.53 |

Accuracies are the grader's re-score at step 100 (its rule also accepts forms such as `#### 18.00`).
Margins: 0.037 to the defective final (its best evaluation) and 0.036 to the control. The control's
step 0 is 0.001, so fixing the launcher without the rerun fails. A first defective run with the same
eps given as a command-line override ended at 0.492, and one with eps 1e-4 at 0.547 (verl's strict
metric). One recorded
run per side, so these are margins, not a variance estimate.
