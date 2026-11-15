# Canonical issue oracle

Reconciled from the task's frozen evidence and from recorded runs of the same recipe that
differ only in the weight-sync settings.

Expected verdict: **unhealthy**.

## Root cause

- `examples/math/gsm8k_icepop.yaml`: `actor.weight_update_mode: disk`, which `run.sh` does
  not override
  - kind: `hparam`
  - observed: the shared config was switched to disk weight updates as a workaround for a
    vLLM hang (its own comment: "xccl to vLLM hangs (FSDP all-gather waits forever); disk
    reload works"). This run uses SGLang (`rollout.backend: "sglang:d4p1t1"`), where the
    hang does not apply. After every PPO step, the FSDP actor writes the full model, about
    28 GB of bf16 safetensors for Qwen1.5-MoE-A2.7B's 14.3B parameters, under the run's
    checkpoint root (`.../weight_update`), and all four SGLang servers reload it from disk.
  - expected: `actor.weight_update_mode: xccl`, the NCCL broadcast this recipe used before
    the switch.

The weight sync becomes the dominant cost of every step. Median `timeperf/update_weights`
is 144.4 s, while `timeperf/train_step` is 20.3 s. The 87-step run needs about 4.2 hours
and is cut off by the job's 4-hour wall-clock limit after 65 steps. A passing diagnosis
names the disk weight-update mode inherited from the shared YAML (a vLLM workaround applied
to an SGLang run) and ties it to the `update_weights` timing. The same YAML change also
raised `rollout.request_timeout` / `actor.request_timeout` to 10800 s, which keeps the slow
reloads from erroring. That masks the problem but does not cause it.

## Observable impact

- Primary source metric: `timeperf/update_weights` in
  `run/experiments/logs/user/kt-seeds/tis_c5-s1/main.log`:
  - 154.3 s at step 0, median 144.4 s, range 139.0 to 290.0 s, above 100 s on all 65
    logged steps;
  - against `timeperf/train_step` median 20.3 s, `recompute_logp` mean 6.2 s, and
    `rollout` mean 0.7 s (rollouts are asynchronous).
- Median wall time per step (from the `StatsLogger` timestamps) is 172.5 s. The log covers
  65 of 87 steps over 3.90 hours. It ends while the SGLang servers are still loading shards
  (`Multi-thread loading shards 0/1` in `run/logs/seeds_15112943.out`).
- Learning itself is unaffected:
  - `eval-rollout/reward` (GSM8K test) is 0.6304 after epoch 1 and 0.6734 after epoch 2;
  - training reward is 0.837 at step 28 and 0.903 at step 64.
  - The failure is that the fixed budget cannot finish in the allotted time.

## Trigger conditions

- Disk weight updates with a model whose full state dict is tens of GB, synced to several
  inference servers after every step.
- In the recorded control (same recipe and seed; `weight_update_mode: xccl`, 3600 s request
  timeouts):
  - median `update_weights` is 3.6 s, and the median step is 35.0 s;
  - all 87 steps finish in 1.49 hours;
  - `eval-rollout/reward` is 0.6674 / 0.6721 / 0.6609;
  - `train_step` is the same as in this run (median 23.1 s).

## Documentation reconciliation

- The truncated log is a consequence of the throughput defect, not a separate root cause.
  The run is killed by the wall-clock limit, not by an error.
- The run is not unhealthy in what it learns: held-out reward and training reward track
  the control. Calling it a learning or stability failure is a wrong attribution.
- These are not the root cause:
  - "The shared filesystem is slow; write the update files to node-local RAM instead."
    That was tried on the same recipe and seed: with the files in `/dev/shm`, the median
    `update_weights` is still 97.5 s and the median step 135.5 s, 3.9× the xccl step. The
    cost is the per-step full-model serialize-and-reload, not the storage medium.
  - Slow training compute (`train_step` 20.3 s here, 23.1 s in the control).
  - Slow generation.
  - Too few GPUs.
  - The TIS correction (`rejection_sampling`, identical in the control).
  - Checkpointing (`save` mean 5.4 s).
- The 4-hour limit is the budget, not the defect: the same budget suffices with xccl.
