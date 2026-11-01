# Historical reward counterexamples

The workspace contains 20 complete episodes across six global steps:

| Step | 1 | 150 | 175 | 190 | 199 | 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Episodes | 4 | 4 | 3 | 2 | 2 | 5 |

Eight are audited full-reward counterexamples across seven tasks. The other 12
are unaudited context, not healthy controls. All episodes share
`workspace/logs/rollouts/<step>.jsonl`; analysis files contain the same IDs.
Complete retained records, including a historical retry index, are preserved.
The curated share is not a population error rate.

| Group | Episodes | Historical contradiction | Rule weakness |
| --- | ---: | --- | --- |
| Dashboard | 1 | Current-user dashboard lacks requested widgets and board links | Selects another user's seeded dashboard by name |
| New shipment | 2 | Tool calls fail; requested new shipment is not created | Accepts seeded shipment |
| New watchlist | 1 | No successful creation or additions | Accepts seeded watchlist/items |
| Firmware/job | 1 | Only list calls, no requested version/job creation | Accepts seeded entities |
| Remaining devices | 1 | Tools report 3 available slots; answer concludes 2 | Old-session clock and incidental number matches |
| Donation leaderboard | 1 | Requested per-member donation counts are omitted | Presence flags are computed but not enforced |
| Discussion thread | 1 | Catalog calls and tool-listing answer, no creation | Accepts pre-existing thread |

All eight have historical score 1.0. The dashboard case includes archived
relevant-table fixtures and recorded calls. The other seven use recorded calls,
responses and answers to establish non-completion, plus separate schema/seed
fixtures demonstrating verifier loopholes. Seed probes are not historical DB
pairs. No-action cases require explicit new creation, not already-satisfied
updates; ambiguous failed writes are not treated as proof of no change.

Each case has a `.md` transcript and `.json` audit. Only lengthy tool catalogs are
omitted from Markdown; raw JSONL retains them. The hidden
`../../curated_trajectory_manifest.json` indexes the eight IDs and hashes.

```bash
python3 -B evaluations/issue/verifier/resources/tools/curate_reward_trajectories.py --replay
python3 -B evaluations/issue/verifier/resources/tools/probe_dashboard_verifier.py
```

Replay uses temporary fixtures, not a historical training rerun. Time-window
checks use recorded timestamps. Source identity is submitter-confirmed on
2026-10-23; no independent runtime attestation, control or measured error rate
is claimed.
