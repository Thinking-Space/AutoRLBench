# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the root-cause and evidence scores are both above `0.8`. The weighted score is reported but does not decide the pass. The gold
verdict for this task is `unhealthy`. The essential root cause is
`PPO_MINI_BATCH_SIZE=4` in `run_gsm8k_grpo.sh` against a rollout batch of
128 prompts × 8 samples, which makes verl take 32 optimizer updates per
rollout batch instead of 4, so most updates are off-policy: entropy rises,
responses truncate, and the training score and held-out accuracy peak early
and decline below the untrained model.
Calling the run healthy, or naming only the symptoms (rising entropy,
truncation, declining reward) without the update count behind them, earns at
most partial root-cause credit. Attributing it to learning rate alone, KL or
entropy coefficients, the reward, data, prompt or model size is wrong.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
