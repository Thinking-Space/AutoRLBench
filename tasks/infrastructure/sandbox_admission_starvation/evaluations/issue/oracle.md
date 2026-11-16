# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

Every agentic trajectory must pass a sandbox admission gate before its first turn, and the gate is far
narrower than the number of trajectories the rollout producer keeps in flight. In
`workspace/vendor/agent_runtime/env_runtime.py`, `MAX_CONCURRENT_SANDBOX_SETUP` defaults to
`SANDBOX_OPERATION_CONCURRENCY` (32), and `create_and_setup_env` holds one `asyncio.Semaphore` slot for
the whole setup sequence: the remote `create`, the repository upload, the tool-command install, and any
retries together with their backoff sleeps. The producer (`workspace/runtime/fully_async.py`, with
`PUBLIC_ASYNC_MAX_INFLIGHT_GROUPS=32` and 8 samples per group in `workspace/scripts/run_config.json`)
keeps 256 trajectories in flight, so most of them queue on the semaphore before generating a single token.
**The semaphore is the half that logs; the binding constraint is the half that does not.** Every one of
those HTTP calls goes through one process-wide `httpx` pool
(`workspace/vendor/agent_runtime/sandbox_client.py`, `SANDBOX_MAX_CONNECTIONS=20`, keepalive 20), shared
by the admitted setups and by the tool calls of every already-running trajectory. The launcher overrides
neither default. Nothing logs a wait for a connection: that time is charged to whichever call was
waiting, mostly the sandbox `create`, and is visible only by comparing the client-side create duration
with the service's own access log. Measured on the public workload, lifting the admission gate alone is
no improvement (366.7 s against 311.7 s unrepaired) because the queue simply moves into the pool, where
the extra concurrent setups now also compete with running trajectories' tool calls; lifting the pool
reaches 177.8 s and lifting both 137.4 s. A diagnosis that names only the semaphore has identified the
symptom that logs, not the limit that binds. The result is starved inference engines and a rollout that
collects groups far more slowly than the engines could serve them; the defect is in the admission and
transport configuration of the real producer path, not the model, the reward, the optimizer, or the
sandbox service being unhealthy.

## Evidence

- Historical archives (`workspace/logs/historical/`): in both runs, `env_runtime.py`-style warnings
  `sandbox setup waited Xs for semaphore` dominate the rollout window. run-a: 67 printed lines (at least
  378 events counting Ray dedup), wait p50 190 s, p90 432 s, max 449 s. run-b: 83 lines (at least 380
  events), p50 205 s, p90 479 s, max 523 s. Engines are idle rather than saturated: per-engine
  `#running-req` mean 1.6 (run-a) / 1.1 (run-b), KV token usage p50 0.02 / 0.04, and no training step is
  logged in either window. run-a's request routing is already cache-aware (prefix hit rate 82.3%), so its
  starvation cannot be blamed on routing; run-b also has the routing problem (hit rate 30.9%), a confounder.
- Hidden reference: a run identical to run-a except `MAX_CONCURRENT_SANDBOX_SETUP` raised to cover the
  256 in-flight trajectories and a larger HTTP pool has no semaphore-wait line, total running requests of
  about 43 in its first 12 minutes and 65 over the archive, and 8 training steps within the hour.
- Public rerun (`workspace/logs/public_rerun/`, Qwen3.5-9B, 4 engines): `env_setup/*.jsonl.gz` records the same
  queueing on the gate (662 setups over three rollouts, 629 of them waited at least 5 s, semaphore wait
  p50 35.4 s, p90 114.5 s, max 160.0 s, against a remote create time p50 of 18.8 s; the producer log shows
  rollout 0 at 0/16 accepted groups after 30, 60 and 90 s), while the rollout and
  the optimizer still proceed; this is the same mechanism at one-node scale. A hidden control run with only
  `MAX_CONCURRENT_SANDBOX_SETUP=8192` and `SANDBOX_MAX_CONNECTIONS=2048` changed has 760 setups with zero
  wait, rollout 0 and 1 in 99 s and 39 s instead of 235 s and 77 s, and the whole run in 837 s instead of
  987 s.

A diagnosis that names the semaphore default and its scope across the setup HTTP calls, connects it to
the 256 in-flight trajectories from the producer configuration, and supports it with the wait and
engine-idle evidence is correct. Attributing the slowdown to the sandbox service's raw latency, to GPU or
KV-cache capacity, to the reward, to model quality, or to request routing alone is wrong or partial.

## Trigger conditions

- Many agentic trajectories are in flight at once (rollout groups x samples per group).
- Each trajectory's sandbox setup takes a slot on a small semaphore and holds it across several remote
  calls and retries.
- The gate and the HTTP connection-pool limits stay at their small defaults.
