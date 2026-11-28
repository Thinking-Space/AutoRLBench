# Canonical issue oracle

Reconciled from the task's frozen evidence and the staged workspace.

Expected verdict: **unhealthy**.

## Root cause

This run's defect chain has three layers; a complete attribution connects all
three. Naming only one layer is partial.

### 1. Environment layer — sandbox instance TTL misaligned with the agent time budget — kind: `hparam`

`scripts/run_grpo_local_swe.sh` configures two independent timeouts that live
in different layers and were never reconciled:

- the sandbox backend's instance lifetime cap,
  `SLIME_AGENT_SANDBOX_TTL_SEC=120` (enforced by the host-side watchdog in
  `slime/slime/agent/sandbox.py`, which kills the sandbox instance when the
  cap expires);
- the harness-side agent time budget,
  `SWE_AGENT_TIME_BUDGET_SEC=900`.

The cap came from a host-cost decision (`notes.md`, "sandbox cost
optimization") calibrated against the fast-episode median; the budget lives
in the rollout layer env. Because the cap (120s) sits far below the budget
(900s) and below the tail of the actual episode duration distribution
(episodes legitimately take 82-295s: agent work plus the in-sandbox verify),
21% of episodes have their instance killed mid-work.

- Observed: `TTL_SEC=120` vs `AGENT_BUDGET=900`; 15 of 72 episodes
  TTL-killed in the frozen run.
- Expected: the sandbox instance cap must cover the whole episode lifetime —
  agent budget plus verify margin — or verify-side infra failures must never
  reach the reward channel as scores (see layer 2).

### 2. Reward-channel layer — every verify-side failure folds into a legal 0.0 — kind: `implementation`

Two folding points, both visible in the staged workspace:

- `examples/coding_agent_rl/swe.py::_LOCAL_VERIFY_WRAPPER` (the
  `tests/test.sh` conversion wrapper written into each sandbox): any
  non-zero checker exit maps to `reward.txt = 0.0` while the wrapper itself
  exits 0, and the checker's stderr is buried in `verify/checker.err`
  instead of propagating. A recycled instance, a missing pytest, a killed
  checker, and a genuinely failing test suite are indistinguishable on this
  path.
- `examples/coding_agent_rl/swe.py::run_local_evaluation`: the reward layer
  wraps upload/exec/readback in a blanket `except Exception` that returns
  `EvalResult(0.0, True)`, writing the traceback only to the per-episode
  `verify/reward_layer.err`. A `SandboxUnavailable` ("instance was recycled")
  therefore scores identically to "the model failed the task".

The recorded run shows the two mechanisms composing: for every TTL-killed
episode the main `run.log` contains only a clean `reward=0.00 ... applied=True`
line, while the first-hand evidence (the `SandboxUnavailable` traceback, the
kill event) is buried in `episodes/<session>/`.

- Observed: infra failures and genuine task failures both produce legal 0.0
  scores that enter training.
- Expected: infrastructure failures are classified (`verifier_infra`-style)
  and excluded from the GRPO group baseline (`remove_sample`), with a
  bounded verifier-only retry; valid zeros stay in training.

### 3. Algorithm layer — false zeros poison the GRPO group baseline — kind: `implementation`

With `n_samples_per_prompt=4`, the false zeros sit inside the group mean that
GRPO subtracts as the baseline:

- every true-score sibling in a polluted group gets its advantage lifted by
  the baseline shift (61% of groups contain at least one false zero);
- each false-zero trajectory eats a full negative advantage even though its
  work was possibly correct (the penalty is proportional to the score it
  would have earned, not to any model error);
- groups that are entirely killed lose all training signal (2 of 9 recorded
  steps had 100% zero rate — signal annihilation).

The kill is not random: the TTL hits exactly the longer, harder-working
trajectories, so the gradient systematically penalizes "working longer",
which is the signature of the reward channel collapsing infrastructure
failure into policy failure. The recorded arm's train reward trajectory
goes DOWN (early=0.159, late=0.055) while the control arm goes UP (early=
0.202, late=0.259) — the TTL is not just adding noise, it is actively
reversing the learning direction.

Listed causes are all essential: the TTL misalignment (layer 1) is the
trigger, the folding chain (layer 2) is what turns it into scores, and the
GRPO pollution (layer 3) is what makes it damage learning. Identifying only
one or two layers must receive partial `root_cause_score`. Do not require
causes beyond these three.

## Observable impact

- Primary metric: per-episode `reward` written by the verify chain and the
  `train/` step metrics computed from it.
- The frozen run's 72 archived episodes: 15 (21%) TTL-killed, every one
  scoring a legal 0.0 with `agent_exit_code=None` (killed mid-work).
- Duration contingency around the 120s cap: `P(reward=0 | duration > TTL)`
  is 0.81 (47 episodes) versus `P(reward=0 | duration <= TTL)` 0.48 (25
  episodes) — the zero score is duration-gated, which no genuine
  model-capability explanation produces.
- GRPO group pollution: 11 of 18 groups (61%) contain at least one false
  zero; 2 of 9 steps had all-zero groups (signal annihilation).
- Train reward trajectory: the recorded arm goes DOWN (early=0.159,
  late=0.055) while the control arm goes UP (early=0.202, late=0.259) —
  the learning direction is reversed, not just slowed.
- Observed train reward: 0.101 (defective) vs 0.193 (control) — the TTL
  cuts the observed reward in half.
- The main `run.log` stays clean throughout: no error strings, `reward=0.00`
  lines indistinguishable from genuine failures — the diagnostics live only
  in the per-episode archives.
- On the healthy configuration (cap=400s, above every episode duration) the
  same seed, data, and stack produce zero kills and 61% non-zero rewards.

## Trigger conditions

- The sandbox backend's TTL cap is below the actual episode-duration tail
  (agent budget 900s, cap 120s).
- The verify chain folds any non-zero checker status and any verify-side
  exception into a legal 0.0 score (wrapper + blanket except).
- GRPO computes the group baseline over the folded zeros.

## Documentation reconciliation

- The recorded arm's observed reward did not crash to zero — it went DOWN
  over steps while the control went UP; "training did not improve" is a
  wrong attribution — the defect is "reversed learning direction with
  halved reward level", not complete collapse.
- "The agent is too weak / the tasks are too hard" is a wrong attribution:
  the control arm earns 61% non-zero rewards on the same data.
- "The model learned to be more efficient" is a wrong attribution for the
  observed duration trend: the duration-gated zero rate (0.81 vs 0.48)
  shows the zeros are infra-caused; a genuine speedup would not be
  TTL-gated.
- "Agent time-budget truncation produces normal zeros" is a wrong
  attribution: the budget (900s) never binds (episodes are 82-295s); the
  kills are TTL-only, and killed episodes carry `agent_exit_code=None`.
- "The reward function / hidden tests are wrong" is a wrong attribution: on
  the aligned configuration the same verify chain scores correctly
  (61% non-zero, zero false zeros).
- The AGS sandbox backend, the vendored fork layout, `qwen25` tool parsing,
  colocated 2×H100 topology, and the per-episode archive design are expected
  properties of this stack, not defects. The TTL watchdog mechanism itself
  is legitimate; only its configuration relative to the episode lifetime is
  the defect.
- A diagnosis that names the TTL misalignment, the two-layer folding, and
  the GRPO baseline pollution, each with its causal role, receives full
  `root_cause_score`.
