# Canonical issue oracle

Expected verdict: **unhealthy**. The rollout producer is starved by admission and transport throttles before trajectories can generate normally; the model, reward, optimizer, and sandbox service are not the primary failure.

## Essential root-cause groups

### 1. Narrow admission gate held across setup

`workspace/vendor/agent_runtime/env_runtime.py` defaults `MAX_CONCURRENT_SANDBOX_SETUP` from `SANDBOX_OPERATION_CONCURRENCY` (32). `create_and_setup_env` holds one semaphore slot across the remote sandbox create, repository upload, tool installation, and retries/backoff. The producer keeps 32 groups with 8 samples each in flight (`PUBLIC_ASYNC_MAX_INFLIGHT_GROUPS=32`), or 256 trajectories, so many trajectories wait on the gate before their first token. The exact configured values are useful corroboration but are not required if the diagnosis correctly explains the undersized gate, its scope, and the resulting queue.

### 2. Shared HTTP pool co-binds the same path

All setup and tool calls use one process-wide `httpx` client in `workspace/vendor/agent_runtime/sandbox_client.py`. Its default connection and keepalive limits are small (20); the launcher changes neither. Setup requests consume those connections while already-running trajectories need them for tool calls. Thus connection-pool contention is a second binding transport limit. Raising only the semaphore moves the queue into the pool instead of restoring throughput. A diagnosis may state the two limits together as one admission/transport bottleneck, but it must make clear that the HTTP pool is shared and independently constraining.

## Accepted evidence (three interchangeable groups)

A correct diagnosis need not reproduce every number below. One concrete observation from each group is sufficient.

1. **Admission:** semaphore wait warnings or measured wait time; the small default/scope relative to producer in-flight count; or delayed/zero accepted groups in the public rerun. Historical archives show long waits (roughly 190–523 s percentiles/max), and the public rerun has hundreds of setups waiting, with the first valid groups delayed.
2. **Transport:** a limited/shared pool, setup and tool traffic competing for it, or client-side create/tool latency much larger than service-side work. The public access log is consistent with a pool near its cap; the pool itself emits no wait log.
3. **Impact:** engines are idle/underused rather than compute-saturated (low running-request and KV usage), historical windows have no training step, or the public rollout and training wait are inflated by queueing. Public setup queueing and rollout wall time are also valid impact evidence.

Routing degradation in one archive is a confounder, not the sole cause; the cache-aware archive still starves. Raw sandbox latency, GPU/KV capacity, reward saturation, model quality, and request routing alone are not root causes.

## Trigger conditions

- Many agentic trajectories are in flight at once.
- Each trajectory holds a small setup slot over several remote calls and retries.
- The gate and shared HTTP pool remain at their small defaults.

## Documentation reconciliation

The hidden controls are author-side context, not required evidence. Opening both limits removes semaphore waits and reduces rollout wall time; opening only the gate leaves the connection-pool queue. These controls are not part of the scoring denominator.
