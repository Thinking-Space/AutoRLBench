# Diagnosis-first task implementation plan

Goal: the agent judges whether the current experiment has a defect, repairing
only when justified; a healthy starting workspace supports a no-change report.
The approved design is DESIGN.md beside this file.

## Constraints

- Reuse the sealed scene SHA b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac.
- Preserve executed runtime and continuation auditor bytes.
- Keep ground-truth verdicts and author validation outside agent-visible mounts.
- Identical prompts, historical logs and environment for both cases.
- No LLM blind tests and no fabricated new GPU acceptance.

## Tasks

1. Add the hidden report assessment wrapper and focused tests. Read report.md,
   parse the starting-state verdict, compare the hidden expected verdict and
   retain the real fresh continuation check. Require both for success. Cover
   healthy/no-change, unhealthy/repaired, wrong verdict and missing artifacts.
2. Rewrite the neutral agent instructions and workspace notes; move operator
   receipts/metadata out of environment/. Create the healthy companion with a
   neutral, correct loader. Pair the same runtime, logs and scene. Record
   inherited GPU evidence and execute exact-content reads against the captured
   real dataset. Update existing packaging tests and source manifests.
3. Review visible files and the normal/faulty acceptance matrix; run the targeted
   suite and validators. Update the bilingual PR description, commit/push and
   read back the remote result. Preserve all original weights and run receipts.
