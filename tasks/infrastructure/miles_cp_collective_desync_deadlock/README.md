# Context-parallel ranks wait on different collectives

Training MiniCPM5-2.6B with GRPO under context parallelism 4; the per-sample log-difference reduction is issued inside a branch that each rank evaluates locally, so a rank with an empty local sample list skips the collective while the others enter it, and the job dies only when the 10-minute watchdog fires.

| | |
| --- | --- |
| **Author** | Bingxu An |
| **Affiliation** | University of Chinese Academy of Sciences |
| **Type** | Infrastructure |
| **Domain** | STEM |
| **Tags** | `context-parallel` `collective` `deadlock` `logdiff` |
| **Framework** | miles (Megatron backend); GRPO; CP=4 |
| **Run time** | crashed after 86 rollout rounds in the recorded job |
| **Resources** | 8 × H800 (80 GB), one node |

## Difficulty

Recorded runs: the same recipe was killed four times at **different rounds** —
`568932` (r86), `569920` (r28), `570089` (r18), `569622` — which is the first
hint that the fault is probabilistic rather than deterministic. The fix landed
one commit later and is a five-line move.

The defect sits in the log-difference metrics of `policy_loss_function`:

```python
if per_sample_max_list:                      # depends on THIS rank's masks
    per_sample_max = torch.stack(per_sample_max_list)
    if args.calculate_per_token_loss:
        if parallel_state.cp.size > 1:
            dist.all_reduce(per_sample_max, op=dist.ReduceOp.MAX,
                            group=parallel_state.cp.group)   # collective, inside
```

`per_sample_max_list` is built from `local_loss_mask_list`. With CP=4 and zigzag
chunking, a rank whose local sample list is empty leaves the list empty and
never reaches `all_reduce`; the other ranks do enter it. A sample with no
response tokens is separately padded with `-inf` by the `elif`; it is the rank
with no samples at all that has no entries. Nobody raises — the ranks simply
wait on different collectives until the watchdog kills the job.

Why it is hard to read off the log:

- the job does not fail at the point of the defect. It **hangs**, and what
  reaches the log 10 minutes later is
  `Watchdog caught collective operation timeout`, i.e. the reporter, not the cause;
- the stack dumps show the ranks blocked in **different collectives on
  different process groups**, which looks like a slow peer unless you notice the
  operations differ;
- the same recipe sometimes runs for 100+ rounds without hitting the condition,
  so a single successful run is not evidence of health.

The strongest distractor is the judge. An earlier investigation of this exact
recipe blamed judge rate limits for the timeouts; the judge is not in the loss
path at all, and the frozen log shows the hang while the judge is healthy.

## Reference solution

- **issue**: a passing diagnosis must name `policy_loss_function`, state that
  the CP `all_reduce` is placed inside a locally-evaluated condition, and
  explain that ranks therefore enter different collectives and desync. It must
  reject "the job crashed", "judge rate limits", "a slow node", and "raise the
  distributed timeout". Naming the mechanism without the local-condition detail
  earns partial credit.

**Tier decision: issue-only.** No solution tier is shipped: all four recorded
arms died, so there is no control run whose metric a threshold could be
calibrated against.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle. The
  canonical judge requires verdict = 1.0 and both root-cause and evidence scores
  > 0.8; the reported score still uses the 0.2 / 0.6 / 0.2 weights.

The frozen evidence is the recorded job's master log, sanitized: absolute paths
and the judge credential/endpoint are replaced, no log line and no metric value
was removed or altered. The package pins the shared `slime-v0.3.2` image for the
issue container; the recorded evidence comes from the internal image, and no
shared-image rerun exists.
