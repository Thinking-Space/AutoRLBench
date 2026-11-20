# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

`miles/backends/training_utils/loss.py::policy_loss_function` — kind:
`collective_desync`

The per-sample log-difference maximum is reduced across context-parallel ranks
with a collective, but that collective sits **inside a condition that is
evaluated locally on each rank**:

```python
if per_sample_max_list:                     # local: depends on this rank's masks
    per_sample_max = torch.stack(per_sample_max_list)
    if args.calculate_per_token_loss:
        if parallel_state.cp.size > 1:
            dist.all_reduce(per_sample_max, op=dist.ReduceOp.MAX,
                            group=parallel_state.cp.group)   # collective, inside the guard
```

With `--context-parallel-size 4` and zigzag chunking, a rank whose local sample
list is empty leaves `per_sample_max_list` empty and therefore never reaches
`all_reduce`, while the other ranks do enter it. A sample with no response
tokens is separately padded with `-inf` by the `elif`; it is the rank with no
samples at all that has no entries. The ranks then wait on different
collectives — a desync, not a slow peer.

- Observed: `dist.all_reduce(...)` reachable only when a rank's local
  `per_sample_max_list` is non-empty.
- Expected: every rank enters the same collective in the same order, with an
  empty local list contributing the identity element (`-inf`, since the op is
  MAX).

## Observable impact

- The run does not raise during training: no Python exception, no OOM, no NCCL
  error at the point of failure. The ranks simply stop making progress inside a
  collective.
- After the 10-minute distributed timeout (`Timeout(ms)=600000`,
  `distributed_timeout_minutes 10`) the watchdog fires and the job is killed, so
  what reaches the log is a timeout, not the cause:
  `Watchdog caught collective operation timeout` plus a dump of the collectives
  each rank was waiting on.
- In the frozen log the two stuck collectives are **different operations on
  different process groups**, which is the signature of a desync: all ranks are
  equally healthy, they are just waiting for different things.
- The same recipe crashed at different rounds across four recorded jobs
  (`568932` r86, `569920` r28, `570089` r18, `569622`), i.e. the fault is
  probabilistic and depends on whether some rank's local sample list is empty.

## Trigger conditions

- `--context-parallel-size` greater than 1 (the recorded run defaulted to 4).
- Zigzag/uneven chunking such that at least one rank's local sample list is empty.
- `--calculate-per-token-loss`, the flag whose branch contains the guarded
  collective.

## Documentation reconciliation

- **"The job crashed" is the wrong diagnosis.** Nothing failed; the ranks are
  alive and blocked in a collective. The timeout is the reporter, not the cause.
- Blaming the judge service is a wrong attribution. An earlier investigation of
  this recipe did blame judge rate limits at first; the judge is not involved in
  the loss path, and the frozen log shows the hang with a healthy judge.
- Blaming a slow node is wrong: unequal-but-finished collectives would still
  complete. Here the two ranks' stack dumps name **different collectives**, so
  no amount of waiting resolves it.
- Raising `--distributed-timeout-minutes` is not a fix: it only delays the kill.
  The flag also does not apply to process groups created without a timeout,
  which is why increasing it from the default changed nothing in practice.
- Switching to the plain (non-logdiff) loss, or setting CP=1, makes the symptom
  disappear but is a workaround, not the root cause.
- A single root cause is present. A diagnosis that says "a collective is
  conditionally skipped" without naming the local condition, CP, and the
  divergent-stack evidence earns partial credit.
