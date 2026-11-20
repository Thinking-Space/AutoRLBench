# Reference repair

`repair.patch` is the reference `scripts/launch_run.sh` as a diff against the staged one. It is
one acceptable repair, not the only one: the verifier grades the outcome of every launch case, on
the staged list and on a held-out list it runs itself, not the shape of the diff.

What the reference changes, in the order the defects bite:

1. **one parse, one source of truth.** The inline validator stays the only reader of the
   manifest, and prints its already-checked values as `KEY=VALUE` lines. The shell reads them
   back into plain variables with a bare assignment — unlike `export VAR="$(...)"`, which returns
   the *export builtin's* status and therefore hides a crashed subshell from `set -e` — and then
   refuses to continue if any expected key is missing.
2. **integer totals.** `as_count()` accepts an int, an integer-valued float or a numeric string
   and refuses anything else with a readable message, instead of `int()` silently truncating
   `300.4` into the run identity.
3. **zero-weight sources are legitimate.** Only the sum has to be positive; sources at count zero
   are excluded from what the data file is expected to contain rather than rejected.
4. **the registry is repo-wide.** Entries whose source this pool does not draw from are skipped
   and counted, not treated as corrupt.
5. **the intended composition is assertable.** `POOL_EXPECTED_TOTAL` and
   `POOL_EXPECTED_SOURCE_COUNTS`, unset by default, are compared with the pool — the one check
   that can tell "this pool agrees with its manifest" apart from "this pool is the experiment I
   meant to run".
6. **the batch configuration is bounded by the pool.** The over-sampling batch may not exceed the
   pool size (otherwise prompts repeat inside a rollout) and may not be smaller than the rollout
   batch it is filtered from.
7. **the validate-only exit runs last**, so a dry run exercises the run identity and the batch
   guards instead of certifying only the pool checks.
8. **the log reports what was observed**, `rows=len(identities)` rather than the manifest's claim,
   and names any registry entries it skipped.

One thing the reference does *not* do: repair a malformed input. A registry with a duplicate
entry, a manifest that contradicts its own data -- those are refusals, not things to clean up on
the way past. Deduplicating the registry instead of refusing it is the one way a complete repair
still fails a held-out case.

Accepted equivalents: a standalone validator script that writes JSON for the shell to read with
`jq`; a validator invoked once per value as long as every value went through the checks; building
the run identity inside the validator. Any of these keeps the decisions right.

Not sufficient, measured on the staged (16) / held-out (22) case lists (`calibration.json`):

| Variant | agreement | unsafe accepts | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: |
| as shipped | 0.375 / 0.545 | 4 / 4 | 6 / 6 | 3 / 3 |
| 1 + 2 (single parse, integer totals) | 0.500 / 0.636 | 2 / 2 | 5 / 5 | 3 / 3 |
| + 3 + 4 (zero weight, registry scope) | 0.688 / 0.727 | 2 / 2 | 5 / 6 | 0 / 0 |
| + 5 (composition assertion) | 0.812 / 0.864 | 2 / 2 | 3 / 3 | 0 / 0 |
| reference | 1.000 / 1.000 | 0 / 0 | 0 / 0 | 0 / 0 |

Repairing the mechanism that explains the empty run tag (1 + 2) still leaves five cases accepted
that should be refused, and the last two unsafe accepts survive until 6 and 7 land. Each pass was
run once per variant per list; the thresholds are exact-match, so they carry no margin by design —
the outcomes are discrete decisions, not a measured metric.
