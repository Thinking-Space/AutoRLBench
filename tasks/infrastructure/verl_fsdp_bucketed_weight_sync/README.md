# FSDP2 parameter grouping drops weights and bucketed sync skips the first chunk

Training Qwen2.5-1.5B-Instruct with GRPO under FSDP2 on Countdown-Tasks-3to4, `build_optimizer` collapses 338 trainable parameters down to 2 by overwriting a `(device, dtype, ndim)` dict, and `BucketedWeightReceiver.receive_weights` calls `on_bucket_received` outside the receive loop so only the last weight bucket ever loads into vLLM. Nothing crashes, `update_weights` reports done every step, and held-out accuracy only moves from 0.0117 to 0.0410 in 45 steps against 0.4658 in the control.

| | |
| --- | --- |
| **Author** | Bo Chen |
| **Affiliation** | University of Hong Kong |
| **Type** | Infrastructure |
| **Domain** | Computer Science |
| **Tags** | `fsdp2` `dtensor` `optimizer-groups` `weight-sync` `zmq-ipc` `rollout-staleness` |
| **Framework** | verl (GRPO, commit `4b9c14f2`), FSDP2 + async colocated vLLM 0.11.0, `run_grpo_countdown.sh` (45 steps, `test_freq=15`) |
| **Run time** | ~27 minutes |
| **Resources** | 1 × A800 (80 GB) |

## Difficulty

At a glance the run looks healthy: `update_weights done` prints every step, `actor/ppo_kl` stays near zero (`-1.05e-6` to `-5.22e-6`), `actor/pg_clipfrac` is `0.0`, and `actor/grad_norm` sits at `0.38–1.07`. Held-out accuracy even ticks up a little (`0.0117` → `0.0410`), which makes it look like slow training rather than a broken update loop.

**Both defects are staged for this task, not upstream verl bugs.** Upstream verl at the pinned commit `4b9c14f2` reads `return optimizer_cls(parameters, **optimizer_args)` in `build_optimizer` and calls `on_bucket_received(weights)` inside the receive loop; `evaluations/solution/oracle/fix.diff` restores exactly those upstream forms. The workspace additionally carries four environment-compatibility changes against that commit (optimizer offload around `optimizer.step()` in `transformer_impl.py`, forced shared-memory weight sync in `vllm_rollout.py`, a benchmark-logger close guard in `tracking.py`, and a benign result-variable refactor in `torch_functional.py`) plus one dropped subdirectory; none of them affect the graded defects.

Two bugs in separate parts of the stack mask each other:

First, `verl/workers/config/optimizer.py::build_optimizer` groups parameters in a dict keyed by `(param.device, param.dtype, param.ndim)`:

```python
param_groups = {}
for param in parameters:
    ...
    key = (param.device, param.dtype, param.ndim)
    param_groups[key] = {"params": [param]}

return optimizer_cls(list(param_groups.values()), **optimizer_args)
```

Under `strategy=fsdp2`, all 338 trainable DTensor parameters share the same device and `fp32` dtype, and every parameter is either 2D or 1D. Each collision overwrites the previous entry, so AdamW only ever receives two tensors: `model.layers.27.mlp.down_proj.weight` (2D) and `model.norm.weight` (1D). The other 336 parameters never update, though backward still runs across the full network and logs normal gradient norms.

Second, `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py::BucketedWeightReceiver.receive_weights` invokes `on_bucket_received(weights)` after breaking out of the `while True:` loop:

```python
while True:
    metadata = self.socket.recv_pyobj()
    weights, tensor = [], None
    ...
    get_torch_device().synchronize()
    self.socket.send(b"")
    if metadata["is_last"]:
        break
on_bucket_received(weights)
```

In `bfloat16`, Qwen2.5-1.5B-Instruct weights total `3,389.5 MiB`, which exceeds verl's default `2,048 MB` bucket cap and splits into two chunks: `bucket[0]` (`2,025.5 MiB`, 214 tensors from `embed_tokens` through `layers.17`) and `bucket[1]` (`1,364.0 MiB`, 125 tensors from `layers.17..27`, `model.norm`, and `lm_head`). The receiver acknowledges `bucket[0]` over ZMQ, resets `weights`, and only passes `bucket[1]` to vLLM.

Because the two parameters surviving `build_optimizer` both sit in `bucket[1]`, actor and rollout weights stay in sync when both bugs are present (`actor/ppo_kl ≈ 0`). Fixing only `build_optimizer` immediately exposes the unsynced first bucket: `actor/ppo_kl` climbs to `0.0159` / `0.0764`, `actor/kl_loss` spikes to `0.7357` / `1.9251`, and accuracy stalls at `0.1650` / `0.2695`. Fixing only `receive_weights` leaves 336/338 parameters frozen (`0.0459` / `0.0664`).

| run | step 0 | step 15 | step 30 | step 45 | `actor/kl_loss` (step 45) | `actor/ppo_kl` (step 45) | `actor/entropy` (step 1 → 45) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.0117 | 0.0146 | 0.0205 | 0.0410 | 0.00207 | -0.00001 | 0.688 → 0.673 |
| defective, seed 43 | 0.0117 | 0.0117 | 0.0156 | 0.0449 | 0.00245 | 0.00003 | 0.649 → 0.619 |
| fix `receive_weights` only, seed 42 | 0.0107 | 0.0137 | 0.0244 | 0.0459 | 0.00464 | 0.00004 | 0.653 → 0.598 |
| fix `receive_weights` only, seed 43 | 0.0117 | 0.0137 | 0.0186 | 0.0664 | 0.00810 | 0.00008 | 0.667 → 0.656 |
| fix `build_optimizer` only, seed 42 | 0.0107 | 0.1260 | 0.1592 | 0.1650 | 0.73572 | 0.01586 | 0.695 → 0.325 |
| fix `build_optimizer` only, seed 43 | 0.0117 | 0.1465 | 0.1553 | 0.2695 | 1.92513 | 0.07638 | 0.690 → 0.154 |
| control, seed 42 | 0.0117 | 0.2402 | 0.4072 | 0.4658 | 0.06364 | 0.00005 | 0.680 → 0.080 |
| control, seed 43 | 0.0107 | 0.2588 | 0.4063 | 0.4883 | 0.08590 | 0.00004 | 0.735 → 0.169 |

Wrong or incomplete diagnoses:
- calling the run healthy, or blaming `lr=2e-6`, `kl_loss_coef=0.001`, async vLLM, Countdown reward parsing, or W&B shutdown errors;
- claiming training did not move at all (the two surviving tail tensors push accuracy from `0.0117` to `0.0410`);
- diagnosing only `build_optimizer` (`0.1650` / `0.2695 < 0.38`) or only `receive_weights` (`0.0459` / `0.0664 < 0.38`).

## Reference solution

- **issue**: name both bugs — `verl/workers/config/optimizer.py::build_optimizer` overwriting `(device, dtype, ndim)` parameter groups down to 2 of 338 FSDP2 tensors, and `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py::BucketedWeightReceiver.receive_weights` calling `on_bucket_received` outside the receive loop so only the last bucket reaches vLLM. Finding only one gets partial `root_cause_score`.
- **solution**: pass `parameters` directly to `optimizer_cls(parameters, **optimizer_args)` in `build_optimizer`, and move `on_bucket_received(weights)` and `del weights, tensor` back inside the `while True:` loop before `self.socket.send(b"")` (`evaluations/solution/oracle/fix.diff`).

## Verification

- **issue**: an LLM judge scores `agent_answer.md` against the hidden oracle on verdict, root cause, and evidence (`0.2 / 0.6 / 0.2`). Passing requires `verdict_score == 1.0` and both other scores above `0.8`.
- **solution**: the agent edits `/workspace`, writes `report.md`, and does not run training. The harness reruns `run_grpo_countdown.sh` (45 steps, 2 h cap) per `run_spec.json`, and `grade_submission.py` re-scores `experiment_logs/val_generations/45.jsonl` with an exact rational AST check (`>= 1,000` rows, pass at `>= 0.38`). Calibrated across 8 runs (2 defective `0.0410` / `0.0449`, 2 receiver-only `0.0459` / `0.0664`, 2 optimizer-only `0.1650` / `0.2695`, and 2 controls `0.4658` / `0.4883` from step-0 `0.0117` / `0.0107`): `+0.1105` above the highest non-control run and `-0.0858` below the lower control. Grader replay exits `1` on all 6 non-control runs and `0` on both controls.
