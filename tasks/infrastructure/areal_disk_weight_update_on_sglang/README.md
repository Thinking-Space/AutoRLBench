# Disk weight sync inherited from a vLLM workaround stalls an SGLang run

Training Qwen1.5-MoE-A2.7B-Chat on GSM8K with asynchronous decoupled PPO (TIS) in AReaL with SGLang rollouts. The shared YAML switched weight sync to disk to work around a vLLM hang, so every step writes and reloads the full ~28 GB model. Median `update_weights` is 144.4 s against 3.6 s with NCCL (xccl), each step takes 172.5 s instead of 35.0 s, and the 87-step run is cut off by its 4-hour limit after 65 steps.

| | |
| --- | --- |
| **Author** | Kaixiang Zhao |
| **Affiliation** | Brigham Young University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `weight-sync` `throughput-regression` `config-drift` `wall-clock-timeout` |
| **Framework** | AReaL v2.0.0 (+33 commits, `9f2967a4`), decoupled PPO with TIS, SGLang 0.5.10.post1 rollouts, FSDP actor |
| **Run time** | 3.9 hours (killed at the 4-hour limit, 65 of 87 steps) |
| **Resources** | 8 × A100-SXM4 (80 GB) |

## Difficulty

The defect is one line of the shared config, `examples/math/gsm8k_icepop.yaml`, which `run.sh` does not override (the switch to disk was made in this workspace's vendored fork of AReaL to work around a vLLM hang; upstream AReaL's `gsm8k_icepop.yaml` still defaults to `xccl`):

```yaml
  weight_update_mode: disk   # xccl to vLLM hangs (FSDP all-gather waits forever); disk reload works
```

The workaround targets vLLM, but this run's rollout backend is `sglang:d4p1t1`. In disk mode, after every PPO step the FSDP actor serializes the whole state dict under the run's checkpoint root (`.../weight_update`), and all four SGLang servers reload it. The same YAML change raised `actor.request_timeout` and `rollout.request_timeout` to 10800 s. Those raised timeouts are why the slow reloads never error out.

Nothing fails loudly:

- the learning curves are healthy: `eval-rollout/reward` is 0.6304 after epoch 1 and 0.6734 after epoch 2, against 0.6674 and 0.6721 for the xccl control, and training reward reaches 0.903 at step 64;
- every worker stays up;
- the log simply stops at step 65, while the SGLang servers are loading shards, when the job reaches its wall-clock limit.

The evidence is in the per-step timing breakdown. Median `timeperf/update_weights` is 144.4 s, ranging from 139.0 to 290.0 s and above 100 s on every step. That is 7× `timeperf/train_step` (median 20.3 s), while `rollout` (mean 0.7 s, asynchronous) and `recompute_logp` (6.2 s) are small.

| | `update_weights` (median) | `train_step` (median) | wall time / step (median) | steps in budget |
| --- | ---: | ---: | ---: | ---: |
| disk to the run's checkpoint root (this run) | 144.4 s | 20.3 s | 172.5 s | 65 / 87 in 3.90 h |
| disk to node-local `/dev/shm` (same seed) | 97.5 s | 29.1 s | 135.5 s | stopped after 7 steps |
| xccl (control, same seed) | 3.6 s | 23.1 s | 35.0 s | 87 / 87 in 1.49 h |

The obvious diagnosis is "the shared filesystem is too slow", and the launcher fix it suggests is wrong. That fix was tried: moving the update files to node-local RAM (`/dev/shm`) brings the update down only to 97.5 s, and the step (135.5 s) is still 3.9× the xccl step (35.0 s). The cost is the per-step full-model serialize-and-reload, not the storage medium. These are also wrong:

- slow training compute, since `train_step` matches the control;
- slow generation;
- too few GPUs;
- the TIS correction, which is identical in the control;
- a learning or stability failure, since held-out reward tracks the control;
- treating the truncated log itself as the defect.

## Reference solution

- **issue**: name `actor.weight_update_mode: disk`, inherited from `gsm8k_icepop.yaml`, as the root cause: a vLLM workaround applied to an SGLang run. Tie it to `timeperf/update_weights` (median 144.4 s, against a 20.3 s `train_step`) and to the run missing its 87-step budget within 4 hours. Blaming storage speed, which leads to the `/dev/shm` remedy, earns partial credit at most. Learning or stability failures, compute, generation or GPU count fail.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.

Issue-only: the failure is wall-clock throughput. A rerun grader would measure time on the evaluator's hardware and filesystem, which the recorded margins cannot calibrate.

Recorded runs, all on 8 × A100 with seed 1 and the same recipe:

- disk weight sync, shipped in `workspace/`;
- disk weight sync with the update files redirected to `/dev/shm`, stopped manually after 7 steps;
- xccl with 3600 s request timeouts, the control.
