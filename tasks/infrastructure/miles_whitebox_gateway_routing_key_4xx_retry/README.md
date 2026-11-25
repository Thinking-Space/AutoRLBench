# Miles white-box gateway drops the routing key and retries deterministic 4xx errors

Training Qwen3-0.6B with GRPO on a dapo-math-17k subset: every rollout request is rejected with 400 missing_routing_key before it reaches the engine, and the retry helper replays that deterministic 4xx for the full 60-attempt budget on both sides of the gateway, so the 30-minute run collects zero samples and completes zero training steps.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | <code>Infrastructure</code> |
| **Domain** | Infrastructure RL · Rollout serving |
| **Tags** | `<http-retry-policy>` `<gateway-header-loss>` `<routing-key>` `<deterministic-4xx>` `<rollout-stall>` |
| **Framework** | miles (GRPO, colocated FSDP actor + sglang engine), sglang 0.5.20.dev59, torch 2.13.0+cu129 |
| **Run time** | ~30 minutes (defective run, stopped at the window; controls finish 10 iterations in ~9) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

Two independent defects compose into the observed stall, and neither fails
loudly:

1. **The boundary defect** (`gateway/whitebox_gateway.py`) — the per-session
   generate route answers through the shared HTTP retry client with
   `headers=None`, so the caller's `X-SMG-Routing-Key` never survives the hop.
   The rollout plugin sends it; the routing proxy rejects every request with
   `400 missing_routing_key`.
2. **The retry defect** (`miles/utils/http_utils.py::_post`, public miles
   history) — every exception, including a deterministic 4xx, is retried for
   the full 60-attempt budget with a 1-second sleep. The 400 therefore burns
   a minute per request before surfacing; the gateway then answers 500, and
   the outer retry budget (the rollout side calls the same helper) restarts
   the whole inner storm: up to 60x60 requests per rollout request.

The strongest evidence against the wrong diagnoses: the identical response
body repeats on every attempt (deterministic, not flaky); the engine never
receives a request (the rejection happens at the routing policy, so "engine
down" is excluded); and the router's error body names exactly the header it
wants (so "router misconfigured" is a symptom reading).

Measured from the recorded arms (single seed per arm; margins, not variance):

| arm | iterations | proxy rejections | inner retry lines |
| --- | ---: | ---: | ---: |
| neither fix | 0 / 20 | 59,520 | 59,520 |
| fail-fast only | 0 / 10 | 10,849 | 0 |
| key forwarding only | 10 / 10 | 0 | 0 |
| both (reference) | 10 / 10 | 0 | 0 |

## Reference solution

- **issue**: a passing diagnosis names both defects with the causal chain
  (dropped routing key at the gateway hop; deterministic-4xx retry on both
  sides of the boundary), and separates them from the symptoms (the proxy's
  rejection, the 500s the gateway answers with) and the wrong diagnoses
  above. Naming only one defect earns partial credit.
- **solution**: two independent repairs — forward the caller's routing key
  across the gateway hop (`headers={"X-SMG-Routing-Key": ...}`), and make
  `_post` raise immediately on a 4xx other than 408/429 while still retrying
  transport errors, 5xx, 408 and 429 (the two lines upstream miles added in
  the commit that follows the vendored pin). Either repair alone is
  insufficient: the key alone leaves the retry defect latent, the fail-fast
  alone still loses the key.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  ≥ 0.8.
- **solution**: the agent submits `report.md` and does not train; the harness
  reruns `bash scripts/run_experiment.sh` (20-minute window, 10 rollout
  iterations) and grades only that run's `experiment_logs/summary.json`:
  pass requires ≥ 5 iterations completed, 0 routing-key rejections, and 0
  gateway inner retry lines (defective side: 0 / 59,520 / 59,520 — margins are
  maximal on every axis; the threshold clears the control's step-0 value
  trivially since the metric is a count of completed iterations). Because a
  key-only repair also completes all 10 iterations, the verifier additionally
  runs sealed checks: the workspace's `_post` must fail fast on
  400/401/403/404/422 and still retry 408/429/500/503 and transport errors,
  and the routing proxy, rollout plugin, recipe, experiment driver, dataset
  and recorded evidence must hash-match the pristine copies. Calibrated on
  one recorded run per arm (2×2 matrix above); the header-only vs both
  distinction was not training-observable by design and is exactly what the
  sealed retry-policy check covers.
