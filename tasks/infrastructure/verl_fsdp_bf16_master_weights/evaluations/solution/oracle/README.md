# Canonical solution oracle

## Root cause

- `run_grpo_gsm8k.sh` (`actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`)
  - kind: `config`
  - observed: bf16 master weights and AdamW state; steps of about 2e-6 round away for most weights,
    the policy drifts slowly and held-out accuracy stays at 0.4625-0.4761
  - expected: `model_dtype: fp32` (verl's default)

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`; `actor/kl_loss` 0.014 at step
  60 vs 0.070 in the control.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `run_grpo_gsm8k.sh`: remove the `model_dtype=bf16` override (`fix.diff`) or set it to `fp32`.

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer, at
least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to cover steps 0-60
and the regenerated training parquet to match the pinned raw GSM8K train split. The runner is not
byte-compared because the repair is in it.
