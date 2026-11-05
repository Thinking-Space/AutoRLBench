# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the
root-cause and evidence scores are both above `0.8`. The weighted score is
reported but does not decide the pass. The gold verdict for this task is
`unhealthy`. The essential root cause is the unmasked length-weighted aggregation in
`workspace/case/loss.py` — `loss = (per_token_loss * lengths).mean()` —
together with `use_remove_padding=false` in
`workspace/case/recorded_config.yaml`, while the shared repaired reference uses
`use_remove_padding=true`: the run differs from the reference in both.
Naming only the custom loss or only the disabled unpadding earns partial
root-cause credit; claiming zero gradients or a reward-side broadcast is
wrong.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
