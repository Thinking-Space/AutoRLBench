# Experiment data-reader workspace

Assess the current implementation before deciding whether to change it. Use
`loader.py`, the prepared scene under `incident/`, and observations under
`logs/recorded/` to investigate. Historical behavior may differ from current
behavior; old errors alone do not establish a current defect.

For solution evaluation, submit `report.md` following the task instruction.
If healthy, leave the code unchanged. If a repair is justified, use the bounded
read-only query `resolve(root, record) -> relative_path` in `loader.py`.
This interface defines the submission boundary, not a claim that it is faulty.

The bundled data preserves 64 training records (32 easy + 32 hard) and 16
evaluation records (8 easy + 8 hard), including original IDs, prompts, answers
and hashes. Difficulty labels are author-supplied historical strata. Prepared
data is read-only; samples cannot be substituted, omitted or modified.

The fixed harness runs fresh CPU data checks after submission and independently
verifies sample ID, actual content hash and difficulty. Locations and catalog
order may change while the original sample contract stays fixed. Both healthy
and repaired submissions face the same checks. No training, model checkpoint,
model/data download or GPU is needed. Use local inspection and non-training
checks; do not launch training or use the network.
