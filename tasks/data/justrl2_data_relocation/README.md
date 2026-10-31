# Missing difficult examples inflate training reward

Native JustRL2 PPO continues training after a stale file mapping silently drops
difficult math examples: file-read aborts jump to 50% and admitted-sample reward
rises from 0.40625 to 0.703125.

| | |
| --- | --- |
| Author | panhaoxuan |
| Domain | data |
| Model | JustRL-II-base-model |
| Dataset | UltraData-RL-2609 / Math (frozen public selection); 64 train, 16 fixed evaluation |
| Framework | miles / JustRL-II, ppo; `run.sh` |
| Environment | `justrl2-data-relocation`; three GPUs |
| Evaluations | issue · solution |
| Recorded runs | One seed per arm: no relocation, stale mapping, updated mapping; eight updates each |

## Difficulty

This is explicitly constructed fault injection using actual model generations
and updates. The sample-ID mapping is in the loader, not a mapping-learning
problem for the model. At the midpoint, hard backing files move while the
manifest retains their old paths. Missing files become ABORTED samples; the
native filter drops them and resamples to fill each batch. Reward still uses the
real math grader, so conventional reward and training-health metrics can look
normal while the admitted population changes.

| Arm | Reward before → after | Post-event abort | Post-event hard share |
| --- | --- | ---: | ---: |
| No relocation | 0.375 → 0.546875 | 0 | 0.5 |
| Stale mapping | 0.40625 → 0.703125 | 0.5 | 0 |
| Updated mapping | 0.390625 → 0.53125 | 0 | 0.5 |

The faulty fixed evaluation falls from 0.5 to 0.3125; both controls also decline.
No repaired-model capability advantage or statistical significance is claimed.
The full records and independent three-arm audit are in the companion proposal.

## Reference solution

The diagnosis must connect the stale mapping, missing-file recovery policy,
resampling, hard-example disappearance, and the changed reward denominator.
The repair must preserve the migration and original examples while publishing
and reading a consistent new mapping. Deleting hard IDs, hiding abort metrics,
or disabling the migration does not repair the data contract.

Only the defective side is staged into `workspace/`. Reference changes and
control evidence are kept under the hidden evaluations; the companion proposal
is not mounted into the agent workspace. Public preparation reconstructs all
66 runtime data files byte for byte from pinned public Math shards. Historical
difficulty assignments are frozen author metadata, not public dataset fields.

## Verification

Issue grading uses the canonical judge: verdict 1.0, root-cause and evidence
scores each greater than 0.8; its reported weighted score uses 0.2/0.6/0.2.
No agent/judge score has been measured for this new task.

For solution evaluation the agent submits `report.md` and does not train. The
harness runs the fixed eight-update command, and the hidden grader independently
checks source identities, physical migration, zero missing-file aborts and 50%
hard coverage after the event, real actor/critic updates, fixed evaluation
snapshots and final optimizer checkpoints. This is a structural integrity grade,
not a noisy accuracy threshold. The baseline hard share is 0 and reference is
0.5; all four post-event steps must satisfy the contract, so initial good data
cannot pass without completing the run.

The pinned parent image, public preparation, fresh native conversion and live
three-GPU training are validated by the companion proposal. The task-specific
linux/amd64 image was built offline and its login-shell/Python smoke check
passed; its receipt is `environment/image_build.json`. Task execution uses the
same pinned parent runtime with the canonical tooling installed. A full agent
harness evaluation, GPU execution of the child image, leaderboard result and
checkpoint restore are not claimed.

The exact packaged preparation and eight-update `run.sh` completed with the
hidden reference patch in job 1428786. Every post-event batch retained 50% hard
examples and zero file aborts. Independent CPU job 1428910 ran the fresh hidden
grader, rehashed both final checkpoint payloads, and returned score 1. The
receipts are recorded in `environment/validation.json` and the hidden oracle.
