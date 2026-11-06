# Checkpoint saving exhausts host memory at step 10

Training Qwen/Qwen2.5-0.5B-Instruct with GRPO on GSM8K. CPU-offloaded
Adam state leaves insufficient host-memory headroom when the native
checkpoint writer stages tensor data for disk at step 10.

| | |
| --- | --- |
| **Author** | Jinyang Wu |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `optimizer-offload` `checkpoint-serialization` `host-memory-exhaustion` |
| **Framework** | verl 0.3.0.post1 (GRPO), FSDP |
| **Run time** | ~3.2 minutes |
| **Resources** | 1 GPU · 20 GiB RAM |

## Difficulty

`actor_rollout_ref.actor.fsdp_config.optimizer_offload` moves Adam state
to CPU after each update. Its moment tensors occupy 3.9GB,
leaving little host RAM for `torch.save` to stage GPU model tensors during
checkpoint serialization. All 10 actor updates complete before Ray's
memory monitor evicts the worker inside `actor_rollout_save_checkpoint`
at its 95% threshold. The cgroup kernel `oom_kill` counter stays at zero;
the recorded exception is host-memory pressure detected by Ray.

`worker_events.jsonl` records `optimizer_state_bytes` at a fixed
instrumentation point in `verl/workers/fsdp_workers.py`: after
`optimizer.step()` and before the post-update offload. The defective run
therefore shows the Adam moments as CUDA-resident at record time; the
offload that moves them to host RAM happens immediately afterward, and
its host-side consequence is visible in `host_memory.jsonl`,
`checkpoint_events.jsonl`, and the Ray eviction — not in the
`optimizer_state_bytes` snapshot.

| Training and checkpoint outcome | Defective | Repaired |
| --- | --- |
| Completed optimizer updates | 10 | 10 |
| Completed trainer-log steps | 9 | 10 |
| Step-10 checkpoint | Ray host-memory eviction | Saved and independently loaded |
| Fresh-process restore | Unavailable | Step 10, validation only |

Validation runs before training and after restoring the checkpoint;
the training loop saves at step 10 without an intervening validation pass.
Defective trainer logs end at step 9 because step 10 is logged after
saving.

## Reference solution

- **issue**: identify CPU-resident Adam state, the additional host-memory
  pressure during native checkpoint serialization and Ray's worker
  eviction. Explain why the trainer log stops before the tenth record.
- **solution**: set
  `actor_rollout_ref.actor.fsdp_config.optimizer_offload: false`, keeping
  optimizer state on the GPU, then complete a fresh 10-step run and save
  the full checkpoint. The reference configuration and calibration
  evidence are hidden in the solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires verdict = 1.0 and both root cause
  and evidence scores > 0.8.
- **solution**: complete a fresh 10-step run within the fixed 20 GiB RAM
  budget, save the full step-10 checkpoint, and restore it in a new process
  for validation without further training. Score: 1.0 if all checks pass,
  otherwise 0.0.

The solution verifier gates the run on a full workspace integrity check:
`protected_source_sha256` pins all 228 workspace files (training driver,
vendored `verl/`, data, frozen logs); only `configs/experiment.yaml` is
editable (semantically compared against `fixed_config`) and only
`report.md` may be added. Modified pinned files or unexpected added files
fail the submission before training starts, so the launch script, Ray
memory-monitor threshold, and vendored framework cannot be altered to
mask the failure. The frozen defective-run evidence must also pass
`check_checkpoint_failure` (10 completed updates, eviction inside
`_save_checkpoint`, no `save_complete`, kernel `oom_kill` zero) as a
pre-flight semantic check of the pinned logs.

## Submission checks

The packaged source and protected-evidence hashes were verified locally.
The frozen defective run passes the checkpoint-failure evidence checks;
the frozen repaired run passes the 10-step update and checkpoint-event checks.
These are offline evidence checks, not a new GPU reproduction. Docker image
building, model preparation, full checkpoint deserialization, and fresh GPU
training/restoration were not rerun during submission. The supplied integration
report records the original defective and repaired runs.

Offline packaging checks (no GPU, standard library only):

```bash
python3 -B -m unittest discover -s evaluations/solution/verifier/tests -v
```

The solution verifier drives training and restoration itself, so no separate
`run_spec.json` is needed. The agent repairs the workspace and submits
`report.md` without launching training.

## Runtime requirements

The 20 GiB cgroup limit is applied by `environment/docker-compose.yaml`,
which only the Harbor backend injects; the native docker runtime passes
`--gpus`/`--shm-size` only and cannot enforce the budget, so run this task
on Harbor. The image is `docker/images/verl_checkpoint_host_memory_oom.Dockerfile`
(pinned `hiyouga/verl` digest, torch 2.5.1 / vLLM 0.7.4 / Ray 2.40.0);
prepare uses the shared `autorlbench_fetch_hf` tooling with revision,
size, and SHA-256 pins (see `environment/data_manifest.json`).
