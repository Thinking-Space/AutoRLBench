# Training entry silently skips HF checkpoint conversion when ray_end.sh exits non-zero

Training Qwen3.5-9B with arena-rubric GRPO using miles. After training completes,
the entry script's tail converts the Megatron checkpoint to HuggingFace format.
This conversion is conditional on `ray_end.sh` exiting 0. `ray_end.sh` copies
`/tmp/ray/` to ceph, but `/tmp/ray/` is absent or empty when the container is
restarted or receives SIGKILL, so `ray_end.sh` exits 1 and the conversion block
is skipped — with RC=1 logged but no visible error surfaced to the user.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Infrastructure |
| **Domain** | Arena-Rubric RL |
| **Tags** | `hf-conversion` `silent-skip` `ray_end` `checkpoint` `post-training` `SIGKILL` |
| **Framework** | miles (GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | ~13 hours (200 rollouts, conversion never happens) |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is in the entry script's tail section — the HF conversion is wrapped in
an `if` conditional:

```bash
RC=$?
if [ ${RC} -eq 0 ]; then
  # ... run ray_end.sh ...
  if bash ray_end.sh; then
    # ... run convert_torch_dist_to_hf ...
  else
    echo "ray_end.sh failed (RC=$?), skipping HF conversion"
  fi
fi
exit ${RC}
```

`ray_end.sh` copies the Ray spill directory (`/tmp/ray/`) to ceph for debugging.
When the container is killed with SIGKILL (not SIGTERM) during training — such as
at a preemption boundary or after the main training process exits cleanly but Ray
workers haven't all flushed — `/tmp/ray/` may be absent or already cleaned up.
In that case `ray_end.sh` exits 1, and the conversion block is silently skipped.

The job exits 0 (training succeeded). The Megatron checkpoint is on disk. But
there is no HF checkpoint, no deployment, and no eval. The user sees "training
complete" and nothing else. The only tell is RC=1 in a log line and the absence
of an `hf-*/` directory at the checkpoint location.

The fix is to delete the conditional entirely and run HF conversion in a separate
job, decoupled from Ray cleanup success.

## Reference solution

- **issue**: identify the `ray_end.sh` exit-status gate on the HF conversion block,
  that SIGKILL leaves `/tmp/ray/` absent causing `ray_end.sh` to exit 1, and connect
  this to the missing HF checkpoint. Reject "training failed" (training succeeded)
  and "SIGKILL is a platform bug" (the fix is not to suppress SIGKILL but to remove
  the conditional).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; fix is removing the conditional and creating
a separate conversion job.
