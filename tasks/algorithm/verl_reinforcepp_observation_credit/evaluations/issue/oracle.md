# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`vendor/verl/verl/trainer/ppo/core_algos.py::compute_reinforce_plus_plus_outcome_advantage` scans backward through response tokens. The defective implementation multiplies `running_return` by `response_mask[:, t]`, so a masked tool observation clears the terminal reward before the scan reaches earlier assistant actions. These internal mask gaps are not episode boundaries. The correct return scan carries future reward across them while retaining the mask for whitening and policy-loss exclusion of observation tokens.

This is one return-estimation defect. The SQL reward function, the use of terminal reward, and masking tool observations out of policy loss are not themselves erroneous.

## Observable impact

- `observed_run/credit_traces.jsonl` retains all 64 rows per estimator call, including rewards, masks, raw returns, whitened advantages, response IDs, and trajectory UIDs. Four rewarded tool trajectories across calls 0 and 2 have terminal reward one. All 2,480 assistant action tokens before the last observation gap have raw return zero.
- Earlier whitened advantages are approximately −0.0920 or −0.1041. Do not equate a zero raw return with zero advantage or zero gradient. Earlier actions can instead be pushed downward by batch normalization.
- Later answer tokens retain terminal reward. The observation mask and return discontinuities distinguish this defect from an all-zero reward or a frozen optimizer.
- `observed_run/tool_traces.jsonl` and rollout records contain real schema/query interactions and submitted final answers. The terminal success must be tied to those records and the corresponding token boundaries; merely finding a generic mask multiplication is insufficient evidence.
- `observed_run/metrics.jsonl` contains three optimizer steps, nonzero rewards and gradients on steps 1 and 3, and zero development accuracy before and after training. Step 2 has all-zero reward. Neither flat development accuracy nor occasional all-zero batches alone identify the defect.

## Trigger conditions

REINFORCE++, a terminal outcome reward, and multi-turn tool rollouts with internal zero-mask observation spans between assistant action spans. A single-turn response without an internal gap does not expose the lost credit. Treating the entire transcript as one unmasked action would corrupt the policy objective and is not an acceptable repair.

## Attribution boundaries

A complete diagnosis must name the accumulator reset across observation masks, explain why it removes earlier action credit, and ground that claim in the recorded terminal reward and return boundaries. Generic sparse-reward, weak-model, SQL-difficulty, truncation, or reward-parser explanations are insufficient.

The hidden exact-batch replay matches defective raw returns and restores earlier reward under the upstream fix; the independent control also carries reward across observations. These controls support attribution but are not required agent-visible evidence. Both short runs remain at zero development accuracy, so neither a learning-performance gap nor broad improvement has been established. Do not attribute all low SQL accuracy to this defect.
