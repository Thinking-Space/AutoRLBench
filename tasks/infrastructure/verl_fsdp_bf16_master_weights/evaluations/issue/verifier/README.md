# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold
verdict for this task is `unhealthy`; the essential root cause is `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` in `run_grpo_gsm8k.sh`, which makes the FSDP actor load and keep its parameters (and AdamW state) in bf16, so steps of about lr = 2e-6 fall below bf16's resolution for most weights and are rounded away.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`, and the raw judge response plus the
request that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
