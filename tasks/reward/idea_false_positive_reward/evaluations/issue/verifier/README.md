# Issue verifier

The standard AutoRLBench judge compares the evaluator-provided
`$AUTORLBENCH_RESULTS/agent_answer.md` with hidden `oracle.md`. It grades diagnosis
only, using verdict/root-cause/evidence weights 0.2/0.6/0.2 and requiring verdict = 1.0, root cause > 0.8, and evidence > 0.8 to pass.

Set your own `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`
in the evaluator environment. They are not needed by the diagnosis agent.
The entry point is `test.sh`; output is `$AUTORLBENCH_RESULTS/result.json`.
No fixed implementation or solution-training tier is included.

Reviewer evidence, provenance and offline replay tools are indexed in
`resources/README.md`. Package regression checks are under `tests/`.
These materials are verifier-side resources, not diagnosis-agent inputs.
