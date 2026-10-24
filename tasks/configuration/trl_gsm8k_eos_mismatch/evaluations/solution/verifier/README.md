# Configuration repair verifier

Run `bash test.sh`, using `AUTORLBENCH_WORKSPACE` and `AUTORLBENCH_RESULTS`
as supplied by the harness. Local testing also accepts `--workspace` and
`--results` arguments. No API credentials, model weights or GPU are needed.

The verifier checks the actual workspace configuration, rejects symlinks,
duplicate JSON keys and invalid token identifiers, preserves non-EOS fields,
and applies the EOS set to synthetic token streams. It writes `result.json`
and `metrics.json` under the results directory and exits unsuccessfully on a
failed check. Results cannot be supplied through agent-authored logs.

This custom verifier is not the official metric-log/retraining grader. A passing
result establishes the requested configuration change, not model-inference
correctness, backend integration or successful retraining. The synthetic stream
checks are deterministic checks of the configuration's intended semantics.
