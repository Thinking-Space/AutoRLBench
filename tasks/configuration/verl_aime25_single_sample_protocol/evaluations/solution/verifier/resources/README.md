# Scoring resources

Entry point: `../test.sh`. The verifier reads
`$AUTORLBENCH_RESULTS/solution_protocol.json` that the rerun stage produced
and checks it deterministically — no LLM, no tolerance to calibrate.

## What is checked

1. **Protocol fields** — the JSON must satisfy all of these:
   - `task == "aime25"`, `benchmark == "aime25"` (case-insensitive)
   - `questions == 30`
   - `samples_per_question == 32`
   - `reducer == "mean"`
   - `temperature == 1.0`, `top_p == 1.0`
   - `expected_generations == 960`
2. **Preserved evidence** — `sha256sum -c` against the oracle manifest
   (`original_eval.sha256`) confirms that nothing under
   `workspace/autorlbench_case/original_eval/` was changed.

Pass is score `1`; any failure is score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace,
submits `/workspace/report.md`, and does **not** run the evaluation.
`run_spec.json` drives the rerun stage: it runs `run_eval.py --protocol-only`
from the workspace root and writes the result to
`$AUTORLBENCH_RESULTS/solution_protocol.json`. This is a CPU-only stdlib call
(a few seconds), so `timeout_s` is set to 60. `log_roots` is empty: no
experiment logs are produced or cleared.
