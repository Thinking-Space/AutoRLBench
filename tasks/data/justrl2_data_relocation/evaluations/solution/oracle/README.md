# Reference repair

The defect is a stale logical sample-ID-to-file-path manifest after a physical
directory migration. Catching FileNotFoundError and retrying another sample
hides the persistent missing hard partition.

Keep the exact midpoint directory move, backing-file bytes, sample IDs and
mathematical answers. Atomically update every hard record's relative path to
`archive/hard/<id>.json` and replace the in-memory mapping used by subsequent
lazy reads. An interrupted update must not leave a partly written manifest.
The reference implementation is the repair branch in the hidden storage file;
it is not mounted in the agent stage.

The solution grade requires completion and data integrity, not raising a model
accuracy metric: zero missing reads and the intended 50% hard mixture after the
event, every real actor/critic update, unchanged source content, all fixed eval
snapshots and final native optimizer checkpoints. Training reward itself may
fall when difficult examples are restored.

Calibration uses one faulty and one repaired eight-update run, with a separate
non-migrated control. Fixed evaluation is small and declines across all arms;
these records establish no repaired-model generalization advantage.
