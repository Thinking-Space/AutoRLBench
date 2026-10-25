# Issue verifier resources

Reviewer-only evidence for 20 complete episodes, including 8 audited full-reward
counterexamples across 7 tasks. These resources are hidden from the diagnosis
agent. Production grading remains `../test.sh`.

| Path | Purpose |
| --- | --- |
| `curated_trajectory_manifest.json` | Current counts, case IDs and evidence hashes |
| `reviewer_notes/curated_reward_examples/` | Eight case transcripts and relevant database/seed fixtures |
| `reviewer_notes/REPRODUCE.md` | Shared-image setup and optional historical training |
| `author_materials/source-provenance.json` | Source revisions, patch references and evidence limits |
| `author_materials/*-worktree.patch` | Captured local source changes, not a reward fix |
| `tools/` | Read-only replay and current container/isolation checks |

## Offline checks

First run `bash environment/prepare.sh` in AutoRLBench's networked prepare
environment. It fetches the pinned dataset; the following checks are offline.
From the prepared task root, using Python 3.11+:

```bash
python3 -B environment/check_assets.py --workspace workspace --manifest environment/data_manifest.json
python3 -B evaluations/issue/verifier/resources/tools/curate_reward_trajectories.py --replay
python3 -B evaluations/issue/verifier/resources/tools/probe_dashboard_verifier.py
python3 -B -m unittest discover -s evaluations/issue/verifier/tests -v
```

The `idea-issue` image provides static inspection dependencies. Replay uses the
standard library, temporary fixture databases and seven reviewed read-only rules.
No check trains, downloads models or calls a hosted judge.

## Evidence boundaries

Steps 1, 150, 175, 190, 199 and 200 contain 4, 4, 3, 2, 2 and 5 episodes. Eight are
audited cases; the other 12 are unaudited context, not healthy controls. This is a
curated subset, not a population error-rate estimate. One case has archived DB
evidence; seven have recorded histories and separately labelled seed-only probes,
not historical DB pairs. See [the case index](reviewer_notes/curated_reward_examples/README.md).

W&B history covers steps 1-199; step 200 is absent and the final stdout record is
truncated. No corrected/control run, completed historical-image build or hosted
semantic judge validation is claimed. Loopback HTTP checks establish transport
only. Standard verdict/root-cause/evidence weights are 0.2/0.6/0.2; passing requires verdict = 1.0, root cause > 0.8, and evidence > 0.8.
