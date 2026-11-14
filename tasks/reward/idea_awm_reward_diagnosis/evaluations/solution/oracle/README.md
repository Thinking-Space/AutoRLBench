# Solution oracle: reference repair of the completion-verifier rules

The seven defective rules are repaired in
[corrected_rules/](corrected_rules/), one file per rule, keyed by
`(scenario, task_idx)`:

| Rule | Scenario : task_idx | Defect | Repair |
| --- | --- | --- | --- |
| `task_management_16_8.py` | task_management_16 : 8 | `SELECT id FROM dashboards WHERE name=?` matches the seed dashboard (id 1, a different user's fully-configured board); the current user's new dashboard is empty | require a **new** dashboard owned by the current user, then run the existing link/widget checks on it |
| `logistics_and_shipment_tracking_1_0.py` | logistics_and_shipment_tracking_1 : 0 | queries the initial shipments but ignores them (the seed already contains the exact requested shipment) | require the matching shipment to be **new relative to the initial state** |
| `stock_trading_1_2.py` | stock_trading_1 : 2 | the seed watchlist 1 already holds exactly the requested five symbols | require a **new** watchlist with the requested items |
| `iot_device_management_3_5.py` | iot_device_management_3 : 5 | the seed already contains the requested firmware version and update job | require a **newly created** firmware version (matching version code, channel, and compatibility) and a **new** update job referencing it |
| `subscription_management_4_2.py` | subscription_management_4 : 2 | the 30-day window is anchored to `MAX(start_time)` (resurrecting stale 2024 sessions) instead of the current date, and the answer check accepts any number token or falls back to unenforced mentions | anchor both window bounds to `now`, then require every stated `N additional devices` claim to equal the computed remaining count (no claims = reject) |
| `fundraising_donations_5_9.py` | fundraising_donations_5 : 9 | computes per-member donation counts/ranks but only enforces name + amount for 80% of members | require, for every member segment, the stated amount **and** a labeled donation count matching the succeeded-only SQL, plus rank |
| `discussion_forum_2_0.py` | discussion_forum_2 : 0 | the seed already contains the requested thread, tags, and post | require a **new** thread with the requested tags and a compliant post |

Each repair keeps the original structure and checks (id-set difference, or
`created_at`-based, or full-row comparison all pass the positives), changing
only the defect.

## Validation

Run from the task root:

```bash
python3 evaluations/solution/oracle/validate_fix.py
```

`validate_fix.py` proves three things (all currently passing):

1. **Corrected rules 14/14** — every negative rejected, every positive accepted.
2. **Original defective rules exactly 7/14** — all seven negatives wrongly
   accepted (the defect reproduced on the exact replay fixtures), while all
   seven positives remain reachable (the cases do not over-constrain).
3. **JSONL round-trip intact** — the corrected codes spliced into a copy of
   the 10,010-entry `gen_verifier.pure_code.jsonl` and reloaded by
   `(scenario, task_idx)` still classify 14/14, i.e. the fix survives the
   delivery format.

The pre-repair copy of the defective file lives outside the task tree at
`outputs/idea-awm-design/gen_verifier.pure_code.jsonl` (design-time staging;
override with `AWM_DEFECTIVE_VERIFIER`). The end-to-end grader behavior was
additionally verified against mock workspaces: unrepaired file → score 0.5 /
exit 1, oracle-spliced file → 1.0 / exit 0, missing file and truncated
entry set → exit 2.

## Fixture-design notes

- The two audited shipment episodes (8b131d98 / 100def7d) share a
  byte-identical fixture set and the shipment rule never reads the final
  answer, so only one negative is retained (documented in
  `resources/replay_cases.json`).
- Positives respect the fixture schemas: `shipments` and `threads` have
  UNIQUE constraints that forbid duplicating the seed rows, so those positives
  are fresh episode states (seed minus the pre-existing row, plus the newly
  created one); timestamps are later than every seed row so creation-time
  comparisons also recognize them as new.
- The leaderboard positive drops the one failed seed donation so the
  succeeded-only SQL and any reasonable tool-side aggregation agree on the
  per-member totals — the correct numbers are unambiguous.
- The firmware positive creates the requested firmware version (the task
  asks to *create a new firmware_version … then schedule a job*, and the seed
  already contains an identical pre-existing row, id 1, so a positive that
  only added a job referencing it would under-specify the request); the new
  job references the newly created firmware. The seed's own firmware row and
  job remain, so the unrepaired rule still reaches the positive via the
  seed job (issue #39).
