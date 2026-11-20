# Maintainer reference

The hidden starting verdict is in `../verifier/resources/expected_verdict.json`.
The original task requires an unhealthy assessment and a corrected query.
`reference_loader.py` is maintainer material; do not expose it to the agent.
The generated healthy control starts with this query and permits an unchanged
submission with a healthy verdict.

The reference reads the current catalog, requires a unique match for both the
requested ID and SHA-256, checks that its backing file exists, and returns its
relative path. It rejects unknown IDs, wrong hashes, missing files and ambiguous
matches. A fixed reader independently verifies payload bytes, ID and difficulty;
the query cannot replace, omit or modify samples.

Every submission must pass the same fresh CPU checks: all 64 original training
records in the bundled scene, then the same IDs and contents after all payloads
move into unpredictable directories with a permuted catalog and wrong-byte
stale-path decoys, plus the unavailable/ambiguous-record probes. The hidden
verifier reruns these checks independently and ignores preplanted JSON or logs.
A correct starting verdict and a nonempty explanation are also required;
solution grading does not judge the prose semantically.

The bundle retains 64 training records (32 easy + 32 hard) and 16 evaluation
records (8 easy + 8 hard) from the frozen UltraData-RL-2609 Math selection, with
original IDs, prompts, answers and hashes. Historical difficulty is
author-supplied, not newly measured. No training or model evaluation is required.

From the task root, demonstrate the repaired reads with:

```bash
python3 -I -S environment/incident_engine/check_reads.py \
  --scene environment/scene \
  --loader evaluations/solution/oracle/reference_loader.py \
  --snapshot environment/snapshot.json
```

This command demonstrates CPU behavior and does not establish isolation. For
an externally produced workspace, use `provenance/evaluate_submission.py` as
shown in [delivery instructions](../../../provenance/DELIVERY.md). It grades
existing submissions in separate CPU Docker prepare/verifier containers with
networking disabled and read-only verifier inputs; it does not launch an LLM
agent. The upstream default Harbor automatic agent workflow does not support
this task's isolated-stage requirement. The existing digest-pinned shared miles
image is retained with `gpus = 0`; its initial pull may need network access.

Original GPU receipts under `provenance/receipts/` are historical only. The
optional full snapshot and continuation code at `aa18e97` are not current
prerequisites.
