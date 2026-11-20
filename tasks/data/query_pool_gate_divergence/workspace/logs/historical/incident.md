# The pool gate is telling us things that are not true

*Notes kept while trying to launch the mixed-pool RL experiment. Written after pass A, the most
recent run of `scripts/run_cases.py` over `cases/cases.jsonl`.*

## What happened

We moved the pool size from a hardcoded number to the manifest, so that one launcher could run
any pool the builder produces. Since then the launch gate has been wrong in both directions,
and pass A (16 cases, 11 accepted, 5 rejected) has an example of each:

* **Pools our own builder makes get refused.** A pool built with one source at weight 0 is
  rejected as having a non-positive source count. So is a single-source pool. So is any pool
  whose manifest came from the older builder. Nothing is wrong with those pools -- `pytest`-free
  as this check is, you can read the data file and count the rows yourself.
* **Runs start on things nobody asked for.** One accepted launch produced the run tag
  `mixed_s50_rebench35_r2e15_grpo_lr3e6_100005` -- note the missing number right after `mixed`.
  Another produced `mixed300.4_...`. A third ran with an over-sampling batch of 256 against a
  120-query pool; its `run.json` records 544 repeated prompts across four rollouts, which means
  the group advantages were computed over duplicated prompts.
* The gate's own log line says `Validated mixed query pool: rows=...` in every one of those
  accepted cases.

## What we checked

* The pool builder is fine: `scripts/build_pool.py` with the same seed produces byte-identical
  files, and every pool under `pools/` matches the digest the case list pins it to.
* The query data is fine: the rows are unique, the tool lists are the four tools we expect, and
  no instance from `registry/missing_images.json` is present in any accepted pool.
* The digest pins work: a corrupted pool is caught.
* The gate is one shell script with one inline Python block. It is not long.

## What is in this directory

```
pass_a/summary.json          the pass's own summary
pass_a/cases.jsonl           one line per case, as the pass recorded it
pass_a/cases/<case_id>/      the launcher output and the run record of 6 cases
```
