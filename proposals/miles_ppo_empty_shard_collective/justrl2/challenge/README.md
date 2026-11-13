# Original empty-shard case: stale distributed PPO statistics

**November 12 redesign:** the injected defect is now a cached-moment identity alias in distributed whitening. It preserves the CP-empty-shard topic while replacing the old rank-local skipped-collective guard. PR #30 stays separate. See [window_validation.json](window_validation.json) for tests, blind review and GPU scope.

The replay completes five updates, but PPO policy metrics first drift after an empty CP shard has participated in a whitening reduction. The first empty update can look correct; the problem appears when the same worker later processes nonempty data. The solver sees only aggregate expected values and normal replay output. It is not asked to add logs or follow a prescribed method.

## Build and run

Use Python 3.12+, CPU PyTorch/Gloo and NumPy with the public JustRL-II checkout at `75814318f243af7e39af337c1173fc3927c9635b`:

    python export_challenge.py --source /path/to/JustRL-II --output /tmp/empty-case
    cd /tmp/empty-case/workspace
    python debug/run_window.py --output /tmp/run-a

Mount only `workspace/` or `workspace.tar.gz`. Keep `maintainer/`, source history and network inaccessible. The exporter pins the upstream source and authored trajectory inputs. The source revision is already repaired; the challenge exporter authors the regression.

## Scenario and computation

The authored replay reuses two copies each of the original trajectories (total/response lengths 424/2 and 357/329). Seeded assignments keep the global multiset, rewards and masks fixed. Under CP2, short responses contribute `[2,0]` local tokens and long responses `[149,180]`.

The defect is in a newly introduced chunked statistics helper used by distributed whitening. It caches a zero-valued identity tensor per device and passes it as the initial value to a lazy reduction over response chunks. When the local response is empty, the reduction returns the identity object itself. The next Gloo all-reduce mutates its input in place. That poisons the cached identity. Later reductions on that worker add the stale three-moment vector (sum, squared sum, valid-token count) to the current batch, even though all ranks enter the same DP collective in the same order.

With the default schedule, earlier mixed updates agree. The empty local partition itself gets the right global moments for its empty output, while contaminating the cached identity. Later nonempty updates produce finite but incorrect policy metrics. A fresh process or a run ending immediately after the empty partition does not show the later discrepancy. This distinguishes a history-dependent local-state error from a permanently wrong process group or normalization formula.

A valid correction must preserve the global sum, squared sum and mask count for each current batch, along with the DP group, unbiased variance correction, GAE, CP slicing and native clipped policy loss. Safe repairs include cloning the reused initializer when the reduction returns it directly or using a fresh zero accumulator for empty input. The native all-zero-global-mask `ValueError` behavior must remain. The replay uses synthetic critic values and log-ratios with native CPU/Gloo operations; no model, optimizer, GPU, or full production sampler is claimed.

## Independent validation

`window_reference.py` predicts per-update outputs from a separate scalar implementation. The private evaluator bundles its own runner, loader, fixture and expected values; it does not trust the contestant runner's reported validation. Private wrappers capture reducer inputs/outputs, rank, group and update identity while leaving collectives and numerical calls unchanged.

Controls include a three-update prefix ending immediately after the empty partition, the full five-update schedule, mixed-only assignment, CP1, CP4, a valid rank permutation, a report-every-update schedule, and the reference correction. The four-topology verifier checks the same-group collective sequence, empty and zero-mask behavior, and numerical output after preceding scenarios have changed reducer state.

The independent blind solver first suspected a wrong process group. Changing the group made results worse; a runtime empty-input reproduction then exposed the reused, in-place-mutated initializer. The report attributes that hypothesis change to execution and validates the repair. One blind run is evidence of meaningful diagnostic difficulty, not a calibrated solve rate or a guarantee that static inspection can never reveal the alias. Historical GPU/restore receipts remain historical and do not validate this authored stateful CPU scenario.
