# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold verdict for this task is `unhealthy`; the essential
root cause is the deferred weight sync (`trainer.rollout_sync_interval=64` in
`run_grpo_stale_sync.sh`, a staged throughput knob), which keeps the resident
rollout replicas sampling from a policy up to 63 steps stale while the
trainer-recomputed `old_log_probs` hide the mismatch from the ratio/clip
metrics, so the drift term distills the policy toward its own stale sampler
and the reward collapses to 0.0 after the step-64 sync. Naming the knob
without the staleness/drift mechanism earns only partial root-cause credit.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`; the raw judge
response and the request that produced it go to
`$AUTORLBENCH_RESULTS/judge-raw.json`.
