# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the root-cause and evidence scores are both above `0.8`. The weighted score is reported but does not decide the pass. The gold
verdict for this task is `unhealthy`. The essential root cause is the
any-number match in `rewards/gsm8k_reward.py::compute_score` (credit whenever
the gold number appears anywhere in the response), which inflates the training
score and the logged validation metric while the policy learns to enumerate
numbers instead of committing to a final answer.
Calling the run healthy, or naming only "the format is not learned" without the
reward's leniency, earns at most partial root-cause credit. Attributing it to
learning rate, KL, data, model size or verl plumbing is wrong.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
