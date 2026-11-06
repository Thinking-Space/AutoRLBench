# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the root-cause and evidence scores are both above `0.8`. The weighted score is reported but does not decide the pass. The gold
verdict for this task is `unhealthy`. The essential root cause is the actor's
AdamW eps override to 1e-3 in `run_gsm8k_grpo.sh`: eps is far above the square
root of the second-moment estimate at this model's per-parameter gradient
scale, so AdamW degenerates into small-step momentum SGD and the run learns
about 30 steps late. Naming only slow learning or a low learning rate earns at
most partial root-cause credit. Calling the run healthy, or blaming sparse
reward, the KL coefficient, the clip ratio, model capacity, the reward or the
data, is wrong.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
