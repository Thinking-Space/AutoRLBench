# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`vendor/verl/verl/trainer/ppo/core_algos.py::compute_reinforce_plus_plus_outcome_advantage` scans backward through response tokens. The defective implementation multiplies `running_return` by `response_mask[:, t]`, so a masked tool observation clears the terminal reward before the scan reaches earlier assistant actions. These internal mask gaps are not episode boundaries. The correct return scan carries future reward across them while retaining the mask for whitening and policy-loss exclusion of observation tokens.

This is one return-estimation defect. The SQL reward function, the use of terminal reward, and masking tool observations out of policy loss are not themselves erroneous.

## Observable impact

The diagnostic evidence is the recorded credit loss and its signatures; quoting
specific rows, token spans, or values is sufficient.

- In the rewarded tool trajectories, assistant action tokens before the last
  observation gap carry raw return zero while the later answer tokens retain the
  terminal reward — the return discontinuity sits exactly at the masked tool
  observations.
- Earlier whitened advantages are approximately −0.0920 or −0.1041. Do not
  equate a zero raw return with zero advantage or zero gradient. Earlier
  actions can instead be pushed downward by batch normalization.
- `observed_run/metrics.jsonl` contains three optimizer steps, nonzero rewards
  and gradients on steps 1 and 3 (batch rewards 2/64, 0/64, 2/64), and zero
  development accuracy before and after training. Step 2 has all-zero reward
  and zero gradient norm. Neither flat development accuracy nor occasional
  all-zero batches alone identify the defect.

## Trigger conditions

REINFORCE++, a terminal outcome reward, and multi-turn tool rollouts with internal zero-mask observation spans between assistant action spans. A single-turn response without an internal gap does not expose the lost credit. Treating the entire transcript as one unmasked action would corrupt the policy objective and is not an acceptable repair.

## Attribution boundaries

A complete diagnosis must name the accumulator reset across observation masks, explain why it removes earlier action credit, and ground that claim in the recorded terminal reward and return boundaries. Generic sparse-reward, weak-model, SQL-difficulty, truncation, or reward-parser explanations are insufficient.

The hidden exact-batch replay matches defective raw returns and restores earlier reward under the upstream fix; the independent control also carries reward across observations. These controls support attribution but are not required agent-visible evidence. Trace-inventory details (the 64 rows per estimator call, the per-token field list, and the four-trajectory accounting across calls 0 and 2), the aggregate count of affected pre-gap action tokens (2,480 across the full sweep), grounding in `observed_run/rollouts/{1,2,3}.jsonl` and their output fields, and record-level linkage between credit-trace token boundaries and rollout records likewise corroborate attribution but are not required evidence items. Both short runs remain at zero development accuracy, so neither a learning-performance gap nor broad improvement has been established. Do not attribute all low SQL accuracy to this defect.
