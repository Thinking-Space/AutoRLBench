# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). A run passes only if the verdict score is `1.0` and the
root-cause and evidence scores are both above `0.8`. The weighted score is
reported but does not decide the pass. The gold verdict for this task is
`unhealthy`. The essential root cause is the post-tool dispatch in
`workspace/vendor/trl/trl/trainer/grpo_trainer.py::GRPOTrainer._generate_single_turn`:
it reuses the initial multiplicity (`self.num_generations`, four) for
continuation generation, while
`workspace/vendor/trl/trl/generation/vllm_generation.py::VLLMGeneration.generate`
subsamples `all_prompts[::num_generations]` and requests `n=num_generations`
outputs from each retained prompt, so after independent tool calls the
returned outputs are assigned to trajectory rows whose own histories were
never sent to the server. Initial generation legitimately has four samples
of a shared prompt; the defect is the retained multiplicity after tool use,
not a global setting that makes four initial samples inherently incorrect.
Blaming sparse rewards, negligible updates, token truncation, or the reward
computation is wrong and earns no root-cause credit.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
