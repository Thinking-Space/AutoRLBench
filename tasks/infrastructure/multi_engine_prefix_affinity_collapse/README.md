# Multi-engine routing destroys prefix-cache affinity

Fully-asynchronous multi-turn GRPO of a MoE policy on sandboxed repository-debugging tasks (public rerun: Qwen3.5-35B-A3B on a synthetic repository workload) asks for per-session routing affinity, but the Miles router it is launched with picks engines by in-flight count only, so consecutive turns of a trajectory land on engines without their prefix cache and the token-weighted prefix hit rate falls from 97.3% to 20.7–30.9%.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Tool-use RL |
| **Tags** | `multi-engine` `prefix-cache` `async-rollout` `throughput` |
| **Framework** | Miles 0.2.1.dev1609+g48f39cb4 (fully-async GRPO), Megatron-Core 0.18.0, SGLang 0.5.15.post1 |
| **Run time** | ~57 minutes (run-b) and ~2.2 hours (run-c) of archived affected window; public rerun 1,044 s for 3 rollouts / 3 optimizer updates |
| **Resources** | Original: 16 × H800 (80 GB), MiniCPM5 MoE policy; public rerun: 8 × H800 (80 GB), Qwen3.5-35B-A3B |

## Difficulty

The original run trained a MiniCPM5 MoE policy with 8 GPUs training and 8 GPUs serving. The public workspace replaces
the private model, prompts, reward service and sandboxes with Qwen3.5-35B-A3B (`Qwen/Qwen3.5-35B-A3B` at
`59d61f3ce65a6d9863b86d2e96597125219dc754`, 4 GPUs expert-parallel training with CPU-offloaded optimizer state, 2
tensor-parallel-2 SGLang engines), 512 synthetic repository-debugging tasks with tool calls whose execution takes
sandbox-like time, and a rule-based reward with a small per-turn efficiency term; the producer, the rollout, the router, the
trainer and the framework source are the real code paths. Nothing fails: every request returns 200 and training proceeds.
The launcher sets `--sglang-router-policy consistent_hashing`, so the rollout gives every sample a routing key and sends it
as `X-SMG-Routing-Key` on every turn, but it also sets `--use-miles-router`, whose `MilesRouter._use_url` chooses the worker
with the fewest in-flight requests and never reads the key.

| Archived run (16 GPUs) | Router | Prefix hit rate (5-min windows) | Total running requests | Gen throughput / engine |
| --- | --- | ---: | ---: | ---: |
| run-a, launcher before the migration | SGLang router | 97.3% (96.5–98.6%) | 42.8 | 398 tok/s |
| run-b, 4 engines | Miles router | 20.7% (2.5–57.2%) | 8.4 | 260 tok/s |
| run-c, 8 engines | Miles router | 30.9% (10.3–51.6%) | 8.6 | 145 tok/s |
| hidden reference, 8 engines | SGLang router, cache-aware | 82.3% | 13.0 | 188 tok/s |

run-b and run-c also carry a second, independent problem: a sandbox admission backlog (semaphore wait p50 224 s / 205 s).
It keeps concurrency low on its own, which is why the hidden reference recovers the hit rate (82.3%) but not the
concurrency; the prefix-cache collapse is specific to the router. The public rerun reproduces the mechanism at one-node
scale with two engines and 256 trajectories in flight; a control run that differs only in using the SGLang router with
`consistent_hashing` keeps sessions on their engine (one run per side):

| Public rerun (Qwen3.5-35B-A3B, 2 engines) | Miles router (buggy) | SGLang router, consistent hashing |
| --- | ---: | ---: |
| Follow-up turns whose cached prefix covers the previous turn's prompt | 61.2% (1,115 / 1,823) | 83.8% (1,539 / 1,836) |
| Token-weighted prefix hit rate of follow-up turns | 48.0% | 56.4% |
| Optimizer updates (grad norm) | 3 (0.70, 0.78, 0.80) | 3 (0.64, 0.78, 0.72) |

With two engines a load-only choice lands back on the right engine about half the time by chance, and Qwen3.5's hybrid
attention does not cache generated tokens, so the public gap is smaller than the archived one; it widens with more engines.
Plausible but wrong diagnoses are KV-cache capacity (KV usage p50 0.04–0.05 in run-b/c), the number of engines (4 and 8
engines both collapse), the sandbox gate alone, the reward and model quality. Each public arm was run once, so the
threshold carries a margin, not a variance estimate.

## Reference solution

- **issue**: name `MilesRouter._use_url`'s in-flight-count selection as ignoring the per-sample routing key that the
  launcher's `consistent_hashing` policy causes the rollout to generate and send; note that the policy only reaches the
  SGLang-router branch; tie it to the prefix-cache hit rate collapse between run-a and run-b/c. Blaming KV capacity, engine
  count, GPU memory, the sandbox waits alone, the reward or model quality fails.
- **solution**: take the Miles router out of the path (`router.use_miles_router` false, the minimal form in
  `evaluations/solution/oracle/repair.patch`), or make `MilesRouter` bind a session to a worker by its routing key.
  Changing `router.policy` alone does not help: the Miles router never reads the policy.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8. `environment/check_evidence.py` verifies the sanitized
  archives (record counts and SHA-256), re-derives every metric file from them, and scans the workspace for private
  markers. `python3 -m autorlbench rerun tasks/infrastructure/multi_engine_prefix_affinity_collapse` was run on one
  8 × H800 node with the task image's cluster mirror and `--backend local`: prepare passed (pinned model, wheels and
  workload digests verified), the run converted the checkpoint, completed 3 rollouts of 16 groups × 4 samples and
  3 optimizer updates (grad norm 0.70, 0.78, 0.80), and exited 0.
- **solution**: the agent repairs the workspace and submits `report.md` without training; the harness reruns the
  launcher per `run_spec.json` and grades only that run. The metric is the share of follow-up turns whose cached prefix
  covers the previous turn's prompt, pass at ≥ 0.75. One run per arm on 8 × H800: 0.612 unrepaired, 0.621 with
  `router.policy` changed but the Miles router kept, 0.838 repaired (`evaluations/solution/oracle/calibration.json`). Grading first requires
  three rollouts each with a successful optimizer update, byte-identical `scripts/` and `runtime/` plus the vendored
  engine-response path that the metric is computed from (`vendor/miles/rollout/engine_client/` and
  `vendor/miles/rollout/sglang_rollout.py`), a pristine model /
  topology / training / rollout configuration, and an intact workload (at least 500 sessions and 1,200 follow-up
  turns). The router implementation (`vendor/miles/router/router.py`) stays editable as the documented alternative
  repair location; in the passing configuration it is out of the request path.
