# Training experiment workspace

Assess the current implementation before deciding whether to change it. Use
`loader.py`, the prepared snapshot under `incident/`, and observations under
`logs/recorded/` to investigate. Historical behavior may differ from the current
implementation.

For the solution evaluation, submit `report.md` following the task instruction.
If no defect requires intervention, leave the code unchanged. If a repair is
justified, the executable submission interface is `loader.py` with
`resolve(root, record) -> relative_path`. This interface is an execution boundary,
not an assertion that the module is defective.

The checkpoint and prepared data are read-only. Sampling, generation, rewards,
training and outcome verification belong to the fixed harness. It runs the
remaining updates after submission. Do not start training yourself.
