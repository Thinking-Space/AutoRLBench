# Canonical issue oracle

Reconciled from the task's frozen evidence (the recorded defective run) and the
canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

This run carries two independent defects; a complete attribution names both.

### 1. `gateway/whitebox_gateway.py::generate` — kind: `routing_header_loss`

The per-session generate route answers through the shared HTTP retry client
instead of forwarding the caller's request, but passes `headers=None`:

```python
output = await post(f"{ROUTING_PROXY_URL}/generate", body, headers=None)
```

The caller (the rollout plugin, `rollout/gateway_generate.py`) sends
`X-SMG-Routing-Key` on every request — with the router policy set to
`consistent_hashing` (`recipes/run_training.py` passes
`--sglang-router-policy consistent_hashing`, and the policy makes
`compute_routing_headers` attach each sample's routing key). The gateway drops
it, and the routing proxy in front of the engine (`routing_proxy/strict_router.py`)
rejects every request with `400 missing_routing_key`:

```
response={"error": {"code": "missing_routing_key", "message": "this routing policy
routes every request by its X-SMG-Routing-Key header; send the header with each request"}}
```

- Observed: the gateway route re-issues the request with `headers=None`, so the
  routing key never survives the hop and every rollout request is rejected
  before reaching the engine.
- Expected: a boundary that answers through the client itself must carry the
  caller's headers across explicitly; the routing key in particular.

### 2. `miles/utils/http_utils.py::_post` — kind: `deterministic_4xx_retry`

The retry helper retries every exception, including `httpx.HTTPStatusError`
raised by `response.raise_for_status()` on a deterministic 4xx, for the full
`max_retries` budget with a one-second sleep between attempts:

```python
        except Exception as e:
            retry_count += 1
            ...
            await asyncio.sleep(1)
            continue
```

There is no distinction between transient errors (connect failures, 5xx, 408,
429) and permanent client errors, so the 400 from the dropped routing key is
replayed 60 times per request before the failure surfaces.

- Observed: a 400 response is retried for the full 60-attempt budget
  (`retrying... (attempt N/60 ... response={"error": {"code": "missing_routing_key"...`),
  ~60 seconds per request, with the identical response body on every attempt.
- Expected: a deterministic 4xx fails fast — the request itself is wrong and
  replaying it only delays the failure the caller needs to see.

## Causal chain (what the logs show)

1. The rollout plugin posts each sample's request to the gateway with the
   routing header attached (`rollout/gateway_generate.py`,
   `headers=compute_routing_headers(args, sample)`).
2. The gateway re-issues the request to the routing proxy with `headers=None`;
   the proxy rejects it with `400 missing_routing_key` — visible as the
   repeated identical response body in `recorded_run/gateway.log.gz` and as
   ~59,520 rejected requests in `recorded_run/proxy` (shim) access log.
3. The inner retry budget (`_post`, 60 attempts, 1/s) replays the 400 for a
   full minute; the exception then escapes the route and the gateway answers
   500 to the caller.
4. The outer retry budget — the rollout side calls the same `post()` helper
   against the gateway — treats the 500 as transient and restarts the whole
   inner storm: up to 60x60 requests per sample, with
   `Error: Server error '500 Internal Server Error' ... retrying ... (attempt N/60,
   url=.../sessions/gw-XXXXXX/generate)` repeating in `recorded_run/train.log.gz`.
5. No rollout request ever completes, so zero samples are collected and zero
   training steps run in the 30-minute window (`recorded_run/train.log.gz`
   shows the run being stopped with no iteration).

## Wrong attributions (must fail)

- "The router is misconfigured" — the proxy's rejection is a *symptom* of the
  missing header, and its error body names the header it wants; the caller
  sends it, and it is lost one hop earlier.
- "The engine is down or overloaded" — the engine never receives a request;
  the rejection happens at the routing policy, before the engine, and the
  identical response body on every attempt marks the error deterministic.
- "Network flakiness / transient errors" — the same exact response repeats
  on every attempt for the full budget; nothing about it is transient.
- "The training loop or reward is broken" — training never got a sample to
  train on; the failure is entirely on the request path.

Partial credit: naming only the retry-policy defect or only the header loss
earns partial credit; the two are independent (either one alone still leaves
the run stuck, for different reasons: without the header loss the engine
serves; without the retry defect the failure surfaces in seconds).
