# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the root-cause and evidence scores are both above `0.8`. The weighted score is reported but does not decide the pass. The gold
verdict for this task is `unhealthy`. The essential root cause is
`ROLLOUT_TOP_P=0.5` in `run_gsm8k_grpo.sh`: training rollouts come from the
top-50% nucleus while the update recomputes probabilities over the full softmax
with no rollout correction, so the unsampled tail is never penalized, entropy
explodes and garbage continuations appear. Naming only the entropy explosion
or the drift from the reference earns at most partial root-cause credit, and so
does listing `top_p` as one cause among several next to a weak KL. Calling the
run healthy, or blaming the KL coefficient, the 512-token cap, the learning
rate, the reward or the data, is wrong.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
