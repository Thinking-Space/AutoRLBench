# Canonical solution oracle

## Root cause

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py::vLLMHttpServer.generate`, full-determinism seed injection
  - kind: `symbol`
  - observed: every request on a replica gets the seed
    `replica_rank + rollout.seed`; with vLLM's batch-invariant mode the
    `n` samples of a prompt are byte-identical, every GRPO group has zero
    reward variance, and every advantage is exactly 0.
  - expected: a per-sample seed that differs between the `n` copies of a
    prompt and is the same when the run is restarted.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`.
- The policy never updates: validation is 0.08719 at every point in both
  defective runs, and `critic/advantages/*`, `actor/pg_loss`,
  `actor/grad_norm` and `actor/kl_loss` are exactly 0 at all 30 steps; 64 of
  64 groups have 8 identical responses at every step.

## Trigger conditions

- `actor_rollout_ref.rollout.full_determinism=True`, `rollout.n > 1` with
  sampling, copies of a prompt on the same replica (always, with one replica).

## Required remediation

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: derive the seed
  per sample. `reference.patch` (written for this task; verl main has no fix)
  uses `(rollout.seed + crc32(f"{priority}:{crc32(prompt token ids)}")) &
  0x7FFFFFFF`, where `priority` is the sample's row in the batch, assigned by
  `AgentLoopManager` with `np.arange(len(batch))` before dispatch. The request
  id cannot be used: vLLM receives a fresh uuid4 per call, so seeding from it
  gives diverse groups but runs that no longer reproduce. Turning full
  determinism off, seeding from a random source, or changing `n` or the
  replica count avoids the defect rather than repairing it; the rerun pins
  full determinism, the base seed, `n` and one replica, and checks
  reproducibility.
- Limit: at this commit `ToolAgentLoop` does not forward `priority` (it reaches
  the server as 0), so the reference patch alone does not separate the `n`
  copies of a prompt in multi-turn tool rollouts. The task's recipe uses the
  single-turn agent loop.

## Machine grading and calibration

Machine grading reads only new `experiment_logs` files and requires all of:

1. `val-core/openai/gsm8k/acc/mean@1 >= 0.4` at step 30 (`score_mode=final`),
   with an accuracy cap of 1.0. The step-30 validation dump is re-graded with
   the verifier's own copy of verl's GSM8K scorer against a protected answer
   key: all 1319 test problems must appear exactly once, answers come from the
   key, and the re-grade must agree with the logged value within 0.002.
2. Group diversity: at each of the 30 training steps the rollout dump must hold
   64 prompts with 8 responses each, and fewer than half of the 64 groups may
   have 8 identical responses.
3. Reproducibility: the rerun repeats the graded command for 3 steps under
   `experiment_logs/repeat/`; its rollout dumps for steps 1, 2 and 3 must
   contain exactly the same rows as the main run's.

Two defective and two repaired runs calibrate the thresholds (data seeds 1 and
2, one GPU). Step 30: defective 0.08719 / 0.08719, repaired 0.7384 / 0.7354.
Maximum over every defective validation point: 0.08719. Minimum over repaired
points from step 5: 0.6543. Repaired step 0: 0.08719 (both), so a fix without
training does not pass. Margins at step 30 are 0.313 above the defective arm
and 0.335 below the repaired arm. Identical groups: 64 of 64 at every step of
both defective runs, 0 of 64 at every step of both repaired runs. The 3-step
dumps of the repaired 30-step run equal those of two separate 3-step runs
byte for byte, and those of a restart with `rollout.seed=43` share no
response. These are observed margins over two seeds per side, not variance
bounds or a claim of cross-environment stability.

`control/` holds the repaired data-seed-1 run's sanitized console history,
W&B history and the step-1 and step-30 dumps; `control-seed2/` and
`defective-seed2/` hold the other runs' console histories.
`historical_evidence_manifest.json` lists the defective data-seed-1 validation
series that the workspace freezes.
