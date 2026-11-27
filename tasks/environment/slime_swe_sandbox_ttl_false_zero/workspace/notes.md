# notes.md · engineering notes (run workspace)

## 2026-11 · sandbox cost optimization

The sandbox backend (`slime/slime/agent/sandbox.py`) keeps every episode's
processes alive on the sandbox host for as long as the agent works. During
the pilot we saw a long tail: most episodes finish within a couple of
minutes, but a few hang around for the whole agent budget (15 min), each
pinning a process tree + workspace. With 4-16 concurrent episodes the box
started running out of headroom on longer steps.

Fix: cap sandbox instance lifetime. `SLIME_AGENT_SANDBOX_TTL_SEC` (see the
watchdog in `sandbox.py`) kills an instance once its TTL expires — the same
thing a cloud sandbox provider does with idle/expired instances. The cap went
into the launch env with a value comfortably above the episode median. We
measured the episode median around ~160s in the pilot, so the cap was set at
120s — most episodes are completely unaffected; only the long tail gets
recycled.

Note: the agent-side time budget is a separate knob
(`SWE_AGENT_TIME_BUDGET_SEC`, still 900s in the launch env) — it lives in
the rollout config, not the sandbox config, so it was not touched when the
cap was added.

Verify chain reminder: the wrapper (`tests/test.sh` → `reward/reward.sh`)
already normalizes any checker outcome into a legal score, and
`run_local_evaluation` in `examples/coding_agent_rl/swe.py` treats
verify-side failures the same way, so a recycled instance just scores 0 —
no crash, no NaN, training keeps flowing. This was checked in the pilot and
looked clean (reward curve keeps improving).
