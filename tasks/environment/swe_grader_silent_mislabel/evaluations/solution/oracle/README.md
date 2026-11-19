# Reference repair

`repair.patch` is the reference `rlgrade/` package as a diff against the staged one. It is one
acceptable repair, not the only one: the verifier grades the labels a pipeline produces, on the
staged corpus and on a held-out corpus it runs itself, not the shape of the diff.

What the reference changes, and why each part is necessary (see `calibration.json` for the
measured effect of each subset):

1. **`command.py`** -- add `-rA` (and `-p no:cacheprovider`) to the grading command, and skip
   required test files that no longer exist instead of letting pytest abort the whole run with a
   usage error. Without this nothing is ever rewarded.
2. **`log_parse.py`** -- parse both the `-rA` summary lines and the `--verbose` progress lines,
   keep the full node id including its path, strip xdist worker prefixes and ANSI codes, split
   the failure message off at bracket depth zero, and never let a later `PASSED` overwrite a
   non-passing status for the same id. Whole-file collection errors (`ERROR tests/x.py`) are
   kept, since they are the run's statement about every test in that file.
3. **`match.py`** -- match full node ids exactly, normalising only the path spelling. The
   substring fallback is removed: it silently answers a question about `test_add` with the
   result of `test_add_duplicate`.
4. **`reset.py`** -- restore the whole test surface before grading: every tracked test module,
   every `conftest.py`, the ini files that can register plugins or add `addopts`, plus removal
   of the files a rollout *added* under a test directory and of stale `.pyc` copies. One
   `git checkout` for the list, with the exit status checked, and a `git status` verification
   afterwards; a restore that cannot be completed makes the record `unscorable` rather than
   graded.
5. **`reward.py`** -- distinguish "the run says this patch is wrong" from "the run does not
   say": a timeout, a pytest internal/usage error, or a required id that was never reported and
   whose file did not fail to collect yields `unscorable` with `reward = None`. A required id
   that is missing *because its file failed to import* stays a failure, since the rollout caused
   it.
6. **`executor.py`** -- keep stderr (collection errors and plugin failures are written there),
   and run the command in its own process group so a timeout kills the test processes it
   spawned rather than leaving them in the checkout.

Accepted equivalents: asking pytest for machine-readable output instead of parsing the human
log (a `--report-log` / junit-xml consumer), restoring the test surface by re-copying from the
pristine snapshot instead of `git checkout`, or running the tests from a directory the rollout
cannot write to. Any of these keeps the labels right; the verifier does not care which.

Not sufficient, measured on the staged / held-out corpora (`calibration.json`):

| Variant | agreement | wrong `resolved` | `unscorable` recall |
| --- | ---: | ---: | ---: |
| as shipped | 0.680 / 0.614 | 0 / 0 | 0.00 / 0.00 |
| `command.py` only | 0.608 / 0.665 | 68 / 46 | 0.00 / 0.00 |
| `command.py` + `reset.py` | 0.892 / 0.873 | 13 / 13 | 0.00 / 0.00 |
| reference | 1.000 / 1.000 | 0 / 0 | 1.00 / 1.00 |

`-rA` alone makes the pipeline able to reward a patch and, in the same step, makes it reward
every rollout that rewrote its own report hook. Adding the test-surface restoration removes
those; what is left are the records whose required ids the suite never reports (bound by the
substring fallback to another test) and the runs that timed out and were reported as a confident
`unresolved`. Each pass was run once per variant per corpus; the thresholds carry a margin, not
a variance estimate.
