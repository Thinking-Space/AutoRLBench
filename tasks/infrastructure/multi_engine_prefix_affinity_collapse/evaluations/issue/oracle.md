# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

Multi-turn trajectories lose engine affinity, so every turn's growing prompt is prefilled again on an
engine that does not hold its prefix in the radix cache. The launcher asks for session affinity: it sets
`--sglang-router-policy consistent_hashing`, which makes
`vendor/miles/rollout/generate_utils/generate_endpoint_utils.py::policy_uses_routing_key` true, so
`vendor/miles/rollout/sglang_rollout.py::generate_and_rm_group` gives every sample a stable
`routing_key` and `runtime/agent_generate.py` sends it as `X-SMG-Routing-Key` on every turn. The launcher
also passes `--use-miles-router`, which replaces the SGLang router with
`vendor/miles/router/router.py::MilesRouter`. Its only selection rule, `_use_url`, picks the worker with the
fewest in-flight requests; it never reads the routing key, and
`vendor/miles/ray/rollout/router_manager.py` copies the policy into the router arguments only on the
SGLang-router branch. The affinity machinery therefore runs but has no effect: a
session's consecutive turns land on whichever engine is least loaded at that moment. In the original runs
the session gateway made the same mismatch (the policy was auto-set to `manual` from
`--use-session-server`, and `vendor/miles/rollout/engine_client/transport.py` forwarded to the gateway with
the key, which proxied through the same Miles router).

## Evidence

- Historical archives (`workspace/logs/historical/`), token-weighted prefix hit rate from SGLang
  `Prefill batch` lines: run-a (launcher before the migration, SGLang router) 97.3% with prompt length p50
  17,986 tokens and total running requests about 43; run-b (Miles router, 4 engines) 20.7% over the archive,
  5-minute windows 2.5% to 57.2%, total running requests 8.4; run-c (Miles router, 8 engines) 30.9%,
  windows 10.3% to 51.6%, total running requests 8.6, per-engine generation throughput 145 tok/s versus
  398 tok/s in run-a. No optimizer step is logged in run-b or run-c.
- Confounder: run-b and run-c also log long `sandbox setup waited Xs for semaphore` waits (p50 224 s /
  205 s). That admission backlog lowers concurrency on its own, but it does not explain the prefix-cache
  collapse; a hidden reference run that keeps the same sandbox gate but uses the SGLang router with a
  cache-aware policy restores the hit rate to 82.3% while its waits remain (p50 190 s).
- Public rerun (`workspace/logs/public_rerun/`, Qwen3.5-35B-A3B, two tensor-parallel-2 engines, 256
  trajectories in flight): `turns/*.jsonl.gz` records `prompt_tokens` and `cached_tokens` for every engine
  call. Only 61.2% of follow-up turns (1,115 of 1,823) find the previous turn's prompt cached, i.e. land back
  on their engine, and the token-weighted hit rate of follow-up turns is 48.0%; per-rollout
  `prefix_cache_hit_rate` is 46–52%. A hidden control run that only swaps in the SGLang router with
  `consistent_hashing` gives 83.8% (1,539 of 1,836) and 56.4%. With two engines a load-only choice is right
  about half the time by chance, so the public gap is smaller than the archived one.

A diagnosis is correct when it names the Miles router's load-only selection as ignoring the per-sample
routing key that the launcher's policy causes to be generated and sent, and ties it to the prefix-cache hit
rate collapse. Blaming KV-cache capacity, the number of engines, GPU memory, the sandbox gate alone, the
reward, or model quality is wrong or partial; the sandbox waits are a second, independent problem.

## Trigger conditions

- Several rollout engines serve a multi-turn workload whose prompts grow turn by turn.
- A routing-key policy makes the rollout emit a stable per-session key.
- Requests go through the Miles router (`--use-miles-router`), which selects by in-flight count only.
- Prefix (radix) cache state is local to each engine.
