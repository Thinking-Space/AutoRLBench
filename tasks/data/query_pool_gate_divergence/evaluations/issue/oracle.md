# Canonical issue oracle

Expected verdict: **unhealthy** -- the gate refuses pools the project's own builder produces, and
starts runs whose configuration and data do not match the pool it validated. Its log line says
`Validated mixed query pool: rows=...` in both failure modes.

## Root cause

**(1) The manifest is read by three separate processes, and only one of the three readings is
checked.** In `scripts/launch_run.sh` the inline validator reads the manifest, then
`export POOL_TOTAL="$(python3 -c '... ["total_queries"]')"` reads it again with a hard subscript,
then the `SOURCE_MIX` echo reads it a third time. The validator treats `total_queries` as
optional (`manifest.get("total_queries", sum(...))`); the `POOL_TOTAL` reader requires it. A
manifest without that key therefore passes validation and crashes the second reader.

**(2) `export VAR="$(...)"` returns the status of the `export` builtin, not of the command
substitution, so `set -e` cannot see the second reader fail.** The crashed read yields an empty
string, and the run proceeds with `POOL_TOTAL=""`. Case `c05_manifest_without_total` is accepted
and its run identity is `mixed_s50_rebench35_r2e15_grpo_lr3e6_100005` -- the pool-size component
between `mixed` and the first `_` is missing -- while `run.json` records `pool_total_env: ""`.
This is the mechanism; everything below is what it hides or what sits next to it.

**(3) `int(manifest.get("total_queries", ...))` truncates instead of refusing.** A manifest
whose total is `300.4` passes, and the float reaches the run identity verbatim:
`c06_non_integer_total` runs as `mixed300.4_...`.

**(4) `if any(count <= 0 ...)` rejects pools the builder legitimately produces.**
`scripts/build_pool.py` records the *requested* mix, so a source at weight zero appears with a
count of zero. Three staged cases are refused for this reason alone -- `c02_zero_weight_source`,
`c03_single_source`, `c04_legacy_manifest` -- and the data files behind them are intact.

**(5) The unusable-instance registry is treated as a per-pool list.** `if source not in
expected_sources` raises `invalid registry entry` for any registry row whose source this pool
does not draw from, although `registry/missing_images.json` is a repo-wide record. This is the
second reason the single-source pools are refused.

**(6) `manifest.get("source_counts") != expected_sources` compares a value with itself.** Once
`expected_sources` stopped being hardcoded it is derived from exactly that key, so the check
cannot fail. It stands where a real check is expected.

**(7) Nothing asserts the pool is the intended experiment.** The gate confirms that a pool agrees
with its own manifest. `c08_composition_not_the_experiment` launches a 60/30/10 pool for a case
that states 50/35/15, and `c13_expected_total_mismatch` launches a 120-query pool for a case that
states 300; both are accepted, because the environment values the cases set are read by nothing.

**(8) The batch configuration is not bounded by the now-variable pool size.**
`c09_oversampling_over_pool` runs an over-sampling batch of 256 against a 120-query pool: the
consumer's record shows `duplicate_prompts_total: 544`, so the group advantages of four rollouts
were computed over repeated prompts. `c15_oversampling_below_batch` runs a rollout batch of 16
filtered from an over-sampling batch of 8.

**(9) The validate-only exit happens before the run identity and the batch configuration are
built.** `c10_validate_only_oversampling` reports "validate-only check complete" for a
configuration a real launch cannot run, so the dry run certifies nothing about the launch.

**(10) The log line reports the manifest's claim, not the observed count.** `rows=` is printed
from `expected_total`, which came from the manifest, so the one number a reader would use to
sanity-check the gate is the number the gate was supposed to verify.

Not at fault: the pool builder (same seed, byte-identical output; every pool matches the digest
its case pins), the digest pins (a corrupted pool is caught -- `c16_pool_digest_mismatch`), the
query data (unique rows, expected tool lists, no unusable instance in any accepted pool), and the
unit tests under `tests/`, which pass because their fixtures never exercise any of the above.

## Evidence

* Pass A (`logs/historical/pass_a/`): 16 cases, 11 accepted, 5 rejected. Four of the eleven
  accepted runs do not match the pool they were validated against, and three of the five
  rejections are pools the builder produced.
* `c05_manifest_without_total`: accepted, run tag `mixed_s50_rebench35_r2e15_grpo_lr3e6_100005`
  with the size component empty, `run.json` `pool_total_env: ""`.
* `c06_non_integer_total`: accepted, run tag `mixed300.4_...`.
* `c09_oversampling_over_pool`: accepted, `run.json` `duplicate_prompts_total: 544` over four
  rollouts of a 120-query pool.
* `c02`, `c03`, `c04`: rejected with `manifest source_counts must be positive` / `invalid
  registry entry`, on pools whose data files are intact and whose digests match.
* Every accepted case's launcher log ends with `Validated mixed query pool: rows=...`.
