# Joint assessment and fresh data-contract grade

The hidden expected verdict describes the starting implementation. `report.md`
must contain that verdict and a nonempty explanation. Solution grading does
not assess prose semantics; the issue evaluation has a separate diagnosis judge.

Score 1 requires the correct starting assessment and every behavioral check:

- Read all 64 original training records through the submitted bounded query.
  The fixed reader checks ID, SHA-256 of actual bytes and difficulty.
- In a temporary scene, move all training payloads into fresh unpredictable
  directories, permute catalog rows and place wrong-byte decoys at stale paths.
  Read and verify the same 64 IDs and original contents again.
- Reject unknown IDs, wrong hashes on existing and missing paths, matching
  catalog entries with missing files, and ambiguous catalog matches.

The verifier independently performs a fresh CPU run of `check_reads.py` using
trusted runtime and scene pins. It does not accept preplanted success JSON,
rerun output claims or historical logs as passing evidence. A correct report
alone cannot pass. Healthy submissions may remain unchanged but face the same
checks. Samples cannot be replaced, omitted or modified.

The scene also preserves the original 16 evaluation records. No model is run
against them. Difficulty labels are historical author-supplied strata, not
new performance measurements. No training, model checkpoint, model/data download
or GPU is required by current acceptance.

Use the task-local `provenance/evaluate_submission.py` to grade a workspace
produced by an external agent. It runs separate CPU Docker prepare and verifier
containers with networking disabled and read-only verifier inputs. It requires
a running Docker daemon, the existing digest-pinned miles image and a new output
directory; obtaining the image initially may need network access. The CLI does
not launch an LLM agent or run the separate issue judge. The upstream default
Harbor automatic agent workflow rejects tasks requiring isolated stages and
does not support this task. A direct checker run demonstrates behavior only.

See [delivery instructions](../../../provenance/DELIVERY.md) for the command and
[security limits](../../../provenance/SECURITY.md) for fixed-sample memorization,
the narrow interface, prose grading limits and trusted-runner assumptions.
Historical GPU receipts under `provenance/` do not validate the current checker
or establish blind LLM success rates.
