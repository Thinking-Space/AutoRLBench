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

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires verdict = 1.0 and both root cause
  and evidence scores > 0.8.

## Runtime requirements

The 20 GiB cgroup limit is applied by `environment/docker-compose.yaml`,
which only the Harbor backend injects; the native docker runtime passes
`--gpus`/`--shm-size` only and cannot enforce the budget, so run this task
on Harbor. The image is `docker/images/verl-vllm0.7.4-torch2.5.1-flashattn2.7.4-cu124-py3.10.Dockerfile`
(pinned `hiyouga/verl` digest, torch 2.5.1 / vLLM 0.7.4 / Ray 2.40.0);
prepare uses the shared `autorlbench_fetch_hf` tooling with revision,
size, and SHA-256 pins (see `environment/data_manifest.json`).
