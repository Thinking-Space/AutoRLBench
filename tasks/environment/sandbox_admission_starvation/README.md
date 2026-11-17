# Sandbox admission gate starves asynchronous rollouts

Fully-asynchronous multi-turn GRPO of a MoE policy on sandboxed repository-debugging tasks (public rerun: Qwen3.5-9B on a synthetic repository workload) keeps 256 trajectories in flight, but a 32–64-slot sandbox setup semaphore held across every remote setup call makes them queue for minutes before their first turn, leaving the inference engines idle.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Tool-use RL |
| **Tags** | `sandbox-admission` `connection-pool` `async-rollout` `throughput` |
| **Framework** | Miles 0.2.1.dev1609+g48f39cb4 (fully-async GRPO), Megatron-Core 0.18.0, SGLang 0.5.15.post1 |
| **Run time** | ~23 minutes of archived starved window (run-a); public rerun 987 s for 3 rollouts / 3 optimizer updates |
| **Resources** | Original: 16 × H800 (80 GB), MiniCPM5 MoE policy; public rerun: 8 × H800 (80 GB), Qwen3.5-9B |

## Difficulty

The original run trained a MiniCPM5 MoE policy (8 GPUs train, 8 GPUs serve) on sandboxed repository tasks. The public
workspace replaces the private model, prompts, reward service and sandbox provider with Qwen3.5-9B (`Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`), 512 synthetic repository-debugging tasks with tool calls, a rule-based reward, and
a local HTTP sandbox control plane; the producer, the environment runtime, the trainer and the framework source are the
real code paths. Nothing fails: every sandbox is eventually created, every request succeeds, and the trainer keeps
stepping in the public rerun. The defect shows up only as waiting. Each trajectory takes a slot on
`MAX_CONCURRENT_SANDBOX_SETUP` (default 32, from `SANDBOX_OPERATION_CONCURRENCY`) and holds it through sandbox create,
upload, tool install and retries, while the producer keeps 32 groups × 8 samples = 256 trajectories in flight and all
setup traffic shares a 20-connection HTTP pool.

| Archived run (16 GPUs, 8 engines) | Semaphore wait p50 / p90 / max | Per-engine running requests | Prefix hit rate | Training steps in window |
| --- | ---: | ---: | ---: | ---: |
| run-a (routing already cache-aware) | 190 s / 432 s / 449 s | 1.6 | 82.3% | 0 |
| run-b (routing also degraded) | 205 s / 479 s / 523 s | 1.1 | 30.9% | 0 |
| hidden reference (gate opened, larger pool) | no wait logged | 5.4 (first 12 min) | 82.0% | 8 within the hour |

Waits are logged only above 5 s, and Ray folds repeated lines, so the printed counts (67 and 83 lines) are lower bounds
(378 and 380 events). The semaphore is only the half of the problem that logs: it shares one 20-connection HTTP pool with
every running trajectory's tool calls, and lifting the gate alone moves the queue into that pool rather than draining it
(366.7 s against 311.7 s unrepaired). The public rerun shows the mechanism at one-node scale, against a control with both
limits lifted; one run per arm:

| Public rerun (Qwen3.5-9B, 8 × H800) | Default (buggy) | Gate and pool raised |
| --- | ---: | ---: |
| Sandbox setups / waited ≥ 5 s | 662 / 629 | 760 / 0 |
| Semaphore wait p50 / p90 / max | 35.4 s / 114.5 s / 160.0 s | 0 s / 0 s / 0 s |
| Remote create p50 | 18.8 s | 13.6 s |
| Rollout 0 / rollout 1 wall time | 235 s / 77 s | 99 s / 39 s |
| Run wall time (3 rollouts, 3 optimizer updates) | 987 s | 837 s |

Plausible but wrong diagnoses are sandbox-service latency (create time is a fraction of the wait), GPU or KV-cache capacity
(engines are idle, KV usage p50 0.02), the reward or model quality, and request routing (run-a is already cache-aware and
still starves). Each public arm was run once, so the threshold carries a margin, not a variance estimate.

## Reference solution

- **issue**: name *both* limits throttling the same path — the setup semaphore in
  `vendor/agent_runtime/env_runtime.py` (default 32, one slot held across the whole remote setup sequence and its
  retries) and the shared 20-connection `httpx` pool in `vendor/agent_runtime/sandbox_client.py`, which logs nothing and
  is the one that actually binds — and connect them to the 256 in-flight trajectories set by the producer
  configuration, with the semaphore-wait and idle-engine evidence. Only the semaphore is logged, so a diagnosis that
  stops there scores half the root cause and falls below the pass mark. Attributing the slowdown to the sandbox service's raw latency, compute capacity, the reward, model quality or
  routing alone fails.
- **solution**: lift the unlogged connection-pool limit, and with it the admission gate — the minimal form is
  `SANDBOX_MAX_CONNECTIONS` / `SANDBOX_MAX_KEEPALIVE_CONNECTIONS` and `MAX_CONCURRENT_SANDBOX_SETUP` in the launcher's
  environment (`evaluations/solution/oracle/repair.patch`); sizing the pool in code and narrowing the semaphore's scope
  are accepted equivalents. Acting on the logged warning alone — lifting the gate — does not clear the threshold.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires a score of at least 0.8; the oracle carries two essential root causes, so naming
  only the logged one cannot reach it. `environment/check_evidence.py` verifies the sanitized
  archives (record counts and SHA-256), re-derives every metric file from them, and scans the workspace for private
  markers. `python3 -m autorlbench rerun tasks/environment/sandbox_admission_starvation` was run on one 8 × H800 node
  with the task image's cluster mirror and `--backend local`: prepare passed (pinned model, wheels and workload digests
  verified), the run converted the checkpoint, completed 3 rollouts of 16 groups × 8 samples and 3 optimizer updates
  (grad norm 1.09, 1.34, 0.98), and exited 0.
- **solution**: the agent repairs the workspace and submits `report.md` without training; the harness reruns the
  launcher per `run_spec.json` and grades only that run. The metric is the summed wall-clock time of the first two
  rollouts, pass at ≤ 200 s. One run per arm on 8 × H800: 311.7 s unrepaired, 366.7 s with the gate alone, 177.8 s with
  the pool alone, 137.4 s with both (`evaluations/solution/oracle/calibration.json`). Grading first requires three
  rollouts each with a successful optimizer update, byte-identical `scripts/` and `runtime/` (including the sandbox
  service), a pristine model / topology / training / rollout configuration, and an intact workload: one sandbox per
  trajectory, create + upload + install before use, tool calls executed in sandboxes, deletion on finish and a low
  environment-error rate.
