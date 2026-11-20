# Third-party notices

The queries in `pools/*/queries.jsonl` are problem statements from the public SWE-smith dataset
(`SWE-bench/SWE-smith`, MIT). Prepare rebuilds every pool from three per-source banks of 1200
statements each, shipped in `environment/sources/query_banks.tar.xz`; the statements are
unmodified except for truncation to 1200 characters.

The three source names (`swe_smith`, `swe_rebench`, `r2e_gym`) stand for the three query families
the original run drew from. The instance ids in `registry/missing_images.json` are ids from the
same public dataset; the registry itself was written for this task.

Everything else in this workspace — `scripts/`, `tests/`, `cases/`, `logs/` — is part of the task
and is covered by the task's Apache-2.0 license.
