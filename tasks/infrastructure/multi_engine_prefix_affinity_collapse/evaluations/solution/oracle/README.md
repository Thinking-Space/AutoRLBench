# Canonical repair

`repair.patch` is the minimal repair: drop `--use-miles-router` from the launcher by setting
`router.use_miles_router` to `false` in `scripts/run_config.json`. The launcher already asks for
`--sglang-router-policy consistent_hashing`, and `vendor/miles/ray/rollout/router_manager.py` copies
that policy into the router arguments only on the SGLang-router branch, so with the Miles router out
of the way the per-sample `X-SMG-Routing-Key` that `sglang_rollout.generate_and_rm_group` already
attaches finally decides placement and a session stays on one engine.

An equally valid repair keeps `--use-miles-router` and makes `MilesRouter` honour the key: hash
`X-SMG-Routing-Key` to a worker and keep that binding, instead of `_use_url`'s
`min(self.worker_request_counts, ...)`. That is a larger change — the header has to survive the proxy
path the router serves — and the grader accepts it on the same metric.

## What does not work

- Changing `router.policy` alone (for example to `cache_aware`). The Miles router never reads the
  policy — `vendor/miles/ray/rollout/router_manager.py` assigns it only on the SGLang-router branch —
  so the change is inert: measured at 0.621 against 0.612 unrepaired.
- More engines, more KV, a larger `server_concurrency`, or a different topology. Those sections of
  `run_config.json` are pinned and compared against the pristine configuration anyway.
- Anything that shortens trajectories or reduces the session count: the grader requires at least 500
  sessions and 1,200 follow-up turns.

## Threshold

`share_of_follow_up_turns_reusing_previous_prompt`, computed from the run's own `turns/*.jsonl`: the
fraction of turns after the first whose `cached_tokens` covers at least 90% of the previous turn's
`prompt_tokens`. Pass at ≥ 0.75. Measured on one run per arm on 8 × H800: 0.612 unrepaired, 0.838
repaired, and 0.621 with the policy changed but the Miles router kept. `calibration.json` records all
three.
