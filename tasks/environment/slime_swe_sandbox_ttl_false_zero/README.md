# Sandbox instance TTL misaligned with the agent time budget folds verify failures into legal zero rewards

Training Qwen3-4B-Instruct-2507 with GRPO (slime, 2×H100 colocated) on 154
code-verified Work-scenario tasks from ClawGym-Bench driven by a claude-code
harness inside per-episode AGS sandboxes; a host-cost instance-lifetime cap
of 120s — never reconciled with the 900s agent time budget — kills 21% of
sandboxes mid-work, and the verify chain folds every such infrastructure
failure into a legal 0.0 that enters the GRPO group baseline, polluting 61%
of groups, halving the observed train reward from 0.19 to 0.10, and reversing
the learning trajectory from UP to DOWN while the main log shows only clean
`reward=0.00` lines.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Environment |
| **Domain** | Work |
| **Tags** | `sandbox-ttl` `reward-channel` `false-zero` `grpo-baseline-pollution` `verify-folding` |
| **Framework** | slime v0.3.2 fork (GRPO; sglang 0.5.15 rollout, megatron training, vendored in workspace) |
| **Run time** | ~45 minutes (9 recorded steps, 72 episodes) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

The defect never fails loudly. Nothing crashes, no NaN, no error strings in
`run.log`: the two folding points (`tests/test.sh` wrapper inside each sandbox,
and `run_local_evaluation`'s blanket `except` in
`examples/coding_agent_rl/swe.py`) are both written as ordinary robustness
code, and the reward they emit is a legal 0.0 indistinguishable from "the model
failed the task". The TTL cap itself is a legitimate cost mechanism whose
value (120s) sits above the fast-episode median, so most episodes look
perfectly healthy.

The strongest evidence against the wrong diagnoses:

- Duration-gated zero rate: `P(reward=0 | duration > TTL) = 0.81` vs
  `P(reward=0 | duration ≤ TTL) = 0.48` — no model-capability explanation
  produces a duration gate at exactly the recycle cap.
- The recorded arm's train reward goes **DOWN** over steps (early=0.159,
  late=0.055) while the control arm goes **UP** (early=0.202, late=0.259):
  the TTL is not just adding noise, it is actively reversing the learning
  direction.
- On the aligned configuration (cap=400s ≥ every episode duration) the same
  seed/data/stack produces zero kills and 61% non-zero rewards, so the verify
  chain and hidden tests themselves are healthy.
- The first-hand diagnostics are buried, not absent: `killed_by_ttl` events
  in `episodes/<session>/sandbox.log`, `sandbox_died` flags in
  `episodes/<session>/episode.json`, and `SandboxUnavailable` tracebacks in
  `episodes/<session>/verify/reward_layer.err`.
- 15 killed episodes span 7 different task instances (124-455s), all with
  `agent_exit_code=None` (killed mid-work, not completed-and-failed).

## Reference solution

- **issue**: A passing diagnosis names the three-layer chain — (1) the sandbox
  instance TTL (`SLIME_AGENT_SANDBOX_TTL_SEC=120` in
  `scripts/run_grpo_local_swe.sh`) sitting far below the agent time budget
  (`SWE_AGENT_TIME_BUDGET_SEC=900`); (2) the two-layer folding of every
  verify-side failure into a legal 0.0 (wrapper + blanket except in
  `examples/coding_agent_rl/swe.py`); (3) the resulting pollution of the GRPO
  group baseline (61% of groups polluted, sibling advantage lift, false-zero
  over-penalty on killed trajectories, reversed learning trajectory). Wrong
  attributions that must fail: "model too weak / tasks too hard" (control arm
  earns 61% non-zero), "agent time-budget truncation" (the 900s budget never
  binds; killed episodes have exit=None), "the model learned to be more
  efficient" (the zeros are duration-gated at exactly the TTL), "broken reward
  function" (the aligned configuration scores correctly).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  ≥ 0.8.
