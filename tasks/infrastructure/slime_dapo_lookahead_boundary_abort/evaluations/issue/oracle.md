# Canonical issue oracle

Expected verdict: **unhealthy**.

The rollout is an infrastructure failure even though its selected batch reached the configured target. The oracle has **three separately scored, essential causes**: 1, 2, and 3. Cause 1 is an ownership-recording defect; cause 2 is the scheduler's shared-error control flow at both lifecycle points; and cause 3 is the cleanup exception path that turns an unacknowledged fenced allocation into the shared boundary error. The acceptance check around lines 161–162 is a related latent branch of the same unowned-error defect; it was not the branch that raised in the observed full-batch rollout. Identifying only part of these three causes must receive partial `root_cause_score`; saying only that the scheduler rethrows boundary errors does not establish the ownership and post-selection discard mechanism.

## Root cause

### 1. Boundary errors lose candidate ownership

`workspace/src/gui_actor_rl/dapo_http.py::record_boundary_failure` appends only a human-readable string to `_gui_dapo_boundary_errors`. The record carries no candidate ordinal, group identity, or flag saying whether the candidate was in the consumed prefix. A cleanup error from a surplus lookahead group is therefore indistinguishable from a failure in a group that supplied training data. The error boundary is a global list rather than a per-candidate lifecycle result.

### 2. The scheduler treats an unowned shared error as fatal at both lifecycle points

`workspace/src/gui_actor_rl/dapo_scheduler.py` checks the shared list in the acceptance loop around lines 161–162 and again in the `finally` cleanup path around lines 218–219. Both checks use a global error list without candidate ownership. The first can block consumption before a ready candidate's prefix membership is decided; in the observed rollout, the second runs after `selected` has reached 32 and discards the full batch when a surplus candidate fails cleanup. Whether a boundary failure should stop training must be decided from candidate ownership / consumed-prefix membership, not from the existence of any string in a global list.

### 3. An incomplete fenced cleanup propagates as a boundary failure

`workspace/src/gui_actor_rl/env_client.py::close_session` starts a fenced `cancel_remaining()` loop while `_allocations` is non-empty. If the loop cannot clear every allocation before `asyncio.wait_for(..., timeout=120)` expires, `TimeoutError` (or an external `CancelledError`) propagates out of the `try`; the `finally` block closes the transport and logs `Env cleanup incomplete; outstanding allocation keys`, but the `return True` after that block is unreachable on this path. `workspace/src/gui_actor_rl/generate_with_gui.py` catches that exception, keeps `cleanup_error`, records the shared boundary failure, and raises `GUI lookahead could not verify environment cleanup`. The defect is therefore the exception-to-global-boundary propagation path; a diagnosis must not claim that this timeout path returns `True` or clears `cleanup_error`.

## Evidence

A complete diagnosis should connect these grouped facts from the frozen run:

1. Rollouts 0–4 each accepted 32 groups with `cleanup_ok = 1`; rollout 5 also reached `groups_accepted = 32` but ended with `cleanup_ok = 0`.
2. Rollout 5 submitted 56 groups and consumed a 39-group prefix: 17 surplus groups were requeued, 14 surplus groups completed, 3 were cancelled, and 53 groups were examined. The three-group submission/examination gap matches the cancellations.
3. The batch became ready after `1630.0474 s`, cleanup then took `145.3386 s`, and total time was `1775.3861 s`.
4. `prefix_acceptance_rate = 0.8205` is 32/39, while `submitted_utilization = 0.5714` is 32/56.
5. The trainer excerpt records `GUI rollout start sample=2889 group=361` followed by `RuntimeError: GUI lookahead boundary failed: ['Environment cleanup failed for sample 2889']`.
6. Comparing all six audit files with that excerpt shows rollout 5 covers group indices 306–358 and contains no 361, placing the failing candidate outside the consumed audit prefix.
7. The effective run configuration targeted 32 groups with eight samples per group, 16 lookahead groups, 384 active episodes, and a 180-second cleanup timeout.

## Wrong attributions

Do not treat `cleanup_s = 145.33863190282136`, lookahead concurrency, or the 180-second timeout as the primary cause; they amplify the exposure window. Do not attribute the abort to reward distribution, zero-gradient groups, or an unfilled batch: the observed rollout accepted 32 groups and reached the batch-ready point. A generic statement that environment service cleanup is flaky is insufficient without the ownership/control-flow and cleanup-exception semantics above.
