# Canonical issue oracle

Reconciled from session-documented evidence (Qwen3.5-9B arena-rubric RL;
evidence is a faithful synthetic reconstruction from documented framework behavior).

Expected verdict: **unhealthy**.

## Root cause

- `run-qwen3_5-9b-arena-rubric.sh::ROLLOUT_NUM_GPUS_PER_ENGINE`
  - kind: `hparam`
  - observed: `ROLLOUT_NUM_GPUS_PER_ENGINE=2`, causing sglang 0.5.9 to run with
    tensor parallelism TP=2 for the Qwen3.5 rollout engine. Due to upstream
    sglang issue 21039, Qwen3.5 produces garbled responses when served with
    TP>1 in sglang 0.5.9. Crucially, the garbled responses arrive as HTTP 200
    with normal JSON structure — no error is raised, no timeout occurs, and
    the driver log shows no warning. The rubric judge receives syntactically
    well-formed but semantically nonsense responses, scores them near 0, and
    training reward stays flat throughout.
  - expected: `ROLLOUT_NUM_GPUS_PER_ENGINE=1`, causing sglang to run with TP=1
    for each rollout engine. With TP=1, Qwen3.5 produces coherent responses and
    the rubric judge can meaningfully score them.

## Observable impact

- Primary source metric: `arena_rubric/score_mean` stays near 0 throughout all
  training steps (expected: starts low and rises over rollouts).
- Rollout responses in the frozen samples contain garbled token sequences despite
  having normal `finish_reason: stop` and within-budget lengths.
- CJK character ratio in rollout responses is near 0 (expected: >0.3 for a
  Chinese question dataset).
- Training runs to completion with no error, no retry, and no worker crash.

## Trigger conditions

- `ROLLOUT_NUM_GPUS_PER_ENGINE` >= 2 for Qwen3.5 architecture.
- sglang version 0.5.x (issue 21039 unpatched in this version).

## Documentation reconciliation

- "The rubric judge failed" is a wrong attribution: the judge receives responses
  and scores them; the low scores reflect genuinely incoherent responses, not a
  judge failure.
- "The model did not converge" is a symptom, not a root cause.
- "HTTP 200 means the rollout succeeded" is misleading: sglang 0.5.9 returns
  HTTP 200 with garbled content for Qwen3.5 at TP>1; the status code is not
  a correctness guarantee for this architecture.
- `ROLLOUT_NUM_GPUS_PER_ENGINE=2` appears intentional for throughput; the defect
  is that this setting is incompatible with Qwen3.5 at this sglang version.
