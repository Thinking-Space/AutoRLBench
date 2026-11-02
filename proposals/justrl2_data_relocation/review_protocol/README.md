# Review first; change code only when justified

The task instruction asks whether the current implementation has a defect. It
does not announce a fault, require a patch, or preselect a root cause. Both cases
accept a report beginning with `Verdict: healthy` or `Verdict: unhealthy` and an
explanation of evidence and action. The verdict describes the code as received.

| Starting code | Submitted verdict | Submitted behavior | Expected outcome |
| --- | --- | --- | --- |
| Correct current resolver | healthy | Unchanged; continuation passes | Pass |
| Correct current resolver | unhealthy | Even if continuation passes | Fail assessment |
| Stale resolver | unhealthy | Repaired; continuation passes | Pass |
| Stale resolver | healthy | Unchanged | Fail |
| Either | Correct | Missing/broken continuation | Fail runtime |

The report gate checks the verdict and requires a nonempty explanation; it is
not a semantic judge of the whole explanation. The issue evaluation retains a
separate diagnosis/evidence judge. Solution score is the conjunction of the
correct starting-state verdict and the existing independent continuation audit.
An unchanged report never substitutes for proof that the experiment works.

Both packages use the same agent instruction, historical logs, scene, model
pins and environment bytes. Their current loader implementations differ. The
historical fault is real, but the healthy starting implementation already handles
it. This controls for agents that label any historical error as a current defect.

Author-only receipts and manifests are under task `provenance/`, not under
`environment/`. The normal Docker/Harbor mounts expose the latter. Hidden verdicts
live under `evaluations/solution/verifier/resources/`, and each issue oracle is
separate. Developers reviewing the public repository can see both cases; agents
must receive an isolated single case through the standard harness. The local
backend is for development and provides no isolation.

`dataset_validation.json` records current read behavior on all 64 real scene
records after rehashing the dataset files. The healthy implementation's executable
AST matches the previously validated GPU reference. Report/score and staging tests
use explicit synthetic continuation fixtures. Existing GPU receipts remain
unchanged and are labeled inherited evidence. No new GPU or LLM-agent run is
claimed for this protocol change, and no agent success rate is reported.
