# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`. The gold
verdict for this task is `unhealthy`; the single essential root cause is
that the `uid` assignment in
`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` sits **after**
`batch.repeat(repeat_times=rollout.n, interleave=True)` instead of before
`self._get_gen_batch(batch)`, which makes every GRPO group a singleton.

There is one root cause, so `oracle.md` carries no partial-credit clause
for naming a subset. It does state the opposite requirement: a diagnosis
that stops at "group size is 1" or "advantages are not normalised"
without locating the uid assignment relative to `batch.repeat` is
incomplete and must not receive full `root_cause_score`. That line exists
because the symptom is easy to read off the logs while the cause is one
indirection away, in `compute_grpo_outcome_advantage`'s silent
singleton-group branch.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
