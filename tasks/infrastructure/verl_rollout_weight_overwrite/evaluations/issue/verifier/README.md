# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the
root-cause and evidence scores are both above `0.8`. The weighted score is
reported but does not decide the pass. The gold verdict for this task is
`unhealthy`. The essential root cause is the initial-checkpoint reload inside `finalize_weights` in
`workspace/case/sync.py`, executed after Actor weights are synchronized into
vLLM, so Actor training proceeds (nonzero grad_norm, rising ppo_kl) but rollout
generation keeps using the initial weights and held-out accuracy stays frozen
at its step-0 level.
Calling the run healthy, or attributing the flat reward to
zero advantages or a broken reward callback, is wrong and earns no root-cause
credit.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
