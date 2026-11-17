# Canonical issue oracle

Reconciled from session-documented evidence (2026-11-17 Cybertron cluster;
evidence is a faithful synthetic reconstruction from documented behavior).

Expected verdict: **unhealthy**.

## Root cause

- `scripts/run_training.sh` — HF conversion conditional on `ray_end.sh` exit status
  - kind: `script`
  - observed: the training entry script wraps the HF checkpoint conversion in a
    conditional: `if bash scripts/ray_end.sh; then ... convert ... fi`. `ray_end.sh`
    copies `/tmp/ray/` to ceph after training; when `/tmp/ray/` is absent (after
    SIGKILL or container restart), `ray_end.sh` exits 1. The conversion block is
    silently skipped with only a log line: `ray_end.sh failed (RC=1), skipping
    HF conversion`. The training job exits 0 (training succeeded). No HF checkpoint
    is produced. No error is surfaced to the user.
  - expected: the HF conversion runs unconditionally after training succeeds,
    decoupled from Ray cleanup success. `ray_end.sh` failure should not gate
    checkpoint availability.

## Observable impact

- Primary symptom: no `hf-*/` directory at the checkpoint location after training completes.
- Driver log contains: `ray_end.sh failed (RC=1), skipping HF conversion`
- Training job exits 0; `cctl job get` shows `Succeeded`.
- The Megatron torch_dist checkpoint is on disk; only the HF conversion is missing.

## Trigger conditions

- Training runs to completion successfully (exit 0 from the training process).
- `/tmp/ray/` is absent at job end, e.g. after SIGKILL during training or container restart.
- `ray_end.sh` exits 1 when the source directory is missing.
- The HF conversion block is guarded by `if bash scripts/ray_end.sh`.

## Documentation reconciliation

- "Training failed" is a wrong attribution: the training process exited 0 and the
  Megatron checkpoint is on disk.
- "SIGKILL is a platform bug" is a wrong attribution: the fix is to decouple HF
  conversion from Ray cleanup success, not to suppress SIGKILL.
- "The Megatron checkpoint is corrupt" is a wrong attribution: it is intact; only
  the HF conversion was skipped.
- The log line `ray_end.sh failed (RC=1), skipping HF conversion` is the tell;
  the job status alone (`Succeeded`) does not reveal the problem.
