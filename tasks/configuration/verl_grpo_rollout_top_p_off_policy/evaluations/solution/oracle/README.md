# Canonical solution oracle

## Root cause

- `run_gsm8k_grpo.sh` (`ROLLOUT_TOP_P=0.5` → `actor_rollout_ref.rollout.top_p=0.5`)
  - kind: `config`
  - observed: training rollouts are sampled from the top-50% nucleus, but `old_log_probs` are
    recomputed over the full softmax and `algorithm.rollout_correction` is disabled, so the update
    treats truncated samples as on-policy. The unsampled tail is never penalized and grows: entropy
    0.26 → 5.06, garbage continuations in about a fifth of training rollouts, held-out accuracy
    0.500 at step 10 and 0.486 at step 100.
  - expected: training rollouts sampled from the full distribution.

## Required remediation

- Set `ROLLOUT_TOP_P=1.0` (`fix.diff`, `code_target/`). Validation already decodes greedily, so this
  changes only the training sampler. Keeping `top_p<1` and enabling a correction for the truncated
  sampler is a different repair that was not recorded; the grader only scores the rerun.

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
pre-baked launcher, `top_p=1.0` inline and the same Ray limits on the command line, so the effective
command is identical). `RESULTS.md` has the side-by-side numbers.

| step-100 held-out | strict accuracy |
| --- | ---: |
| defective | 0.486 |
| control | 0.566 |
| threshold | ≥ 0.53 |

Margins: 0.044 to the defective final (its best evaluation is 0.500 at step 10) and 0.036 to the
control. The control's step 0 is 0.001, so fixing the launcher without the rerun fails. A first
defective run of the same configuration, launched with `top_p=0.5` as a command-line override
rather than in the script, ended at 0.456 (best 0.500 at step 10). One recorded run per side, so
these are margins, not a variance estimate.
