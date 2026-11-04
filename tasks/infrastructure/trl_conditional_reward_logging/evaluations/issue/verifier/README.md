# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold
verdict for this task is `unhealthy`; the essential root cause is that
`src/trl/trainer/grpo_trainer.py::GRPOTrainer._generate_and_score_completions`
iterates each rank's own pending metric keys before invoking collectives while
the reward function emits `answer_parse_rate` only on rank 0, so the ranks
issue different collectives at the same sequence number and crash with a
collective fingerprint mismatch before the first optimizer step.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading
reads `oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`.
