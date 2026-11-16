# Canonical repair

Two limits throttle the same path, and only one of them is logged.

1. `MAX_CONCURRENT_SANDBOX_SETUP` (default 32, from `SANDBOX_OPERATION_CONCURRENCY`) — one semaphore
   slot in `vendor/agent_runtime/env_runtime.py::create_and_setup_env` is held across the remote
   create, the upload, the tool-command install and the retries, while the producer keeps 256
   trajectories in flight. This is the limit that prints `sandbox setup waited Xs for semaphore`.
2. `SANDBOX_MAX_CONNECTIONS` / `SANDBOX_MAX_KEEPALIVE_CONNECTIONS` (default 20) — every sandbox
   operation of every trajectory, setup and tool calls alike, shares one `httpx` pool in
   `vendor/agent_runtime/sandbox_client.py`. Nothing logs a wait for a connection; the time is
   charged to whichever call was waiting, mostly the sandbox create.

`repair.patch` lifts both through the launcher's environment. Equivalent repairs in code are
accepted on the same metric: sizing the pool in `sandbox_client.py`, and narrowing the semaphore to
the single operation that needs serialising (or removing it once the pool is sized).

## The logged limit is a decoy

The warning names the semaphore, and lifting only the semaphore does not help — measured at 366.7 s
against 311.7 s unrepaired, i.e. no better and, within the noise of these single runs, slightly
worse. Admitting all 256 setups at once only moves the queue into the 20-connection pool, where the
new setup traffic now also competes with the tool calls of trajectories that are already running.
The binding constraint is the pool: lifting it alone reaches 177.8 s, and lifting both reaches
137.4 s. A repair that follows the only logged signal therefore fails the threshold.

## What does not work

- Making the sandbox service faster. `runtime/sandbox_service.py` is byte-compared.
- Fewer trajectories in flight, fewer turns, shorter rollouts, or skipping setup steps. The
  `rollout` and producer settings are pinned, and the grader checks one sandbox per trajectory,
  create + upload + install before any tool call, tool calls executed in sandboxes, deletion when a
  trajectory finishes, and a low environment-error rate.

## Threshold

`rollout_0_plus_1_wall_s`, the summed wall-clock time of the first two rollouts, from the run's own
`rollout.*.json`. Pass at ≤ 200 s. One run per arm on 8 × H800: 311.7 s unrepaired, 366.7 s with the
gate alone, 177.8 s with the pool alone, 137.4 s with both. `calibration.json` records all four and
the matching CPU calibration.
