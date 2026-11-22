# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement. As implemented
in `judge.py` (shared verbatim with the other tasks), a run passes only if
`verdict_score == 1.0`, `root_cause_score > 0.8` and `evidence_score > 0.8`;
the weighted score (0.2 / 0.6 / 0.2) is reported in `result.json` but does not
decide the pass. The gold verdict for this task is `unhealthy`; the essential
root cause is that resuming stage 2 from the stage-1 checkpoint restores the
optimizer and LR-scheduler state (`base_lrs = [1e-06]`, param-group `lr`) in
`FSDPCheckpointManager.load_checkpoint`, so the configured `optim.lr=2e-6` is
silently replaced and the run trains at 1e-6 (`actor/lr` 1e-06 at steps 3-5).

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`; the raw judge
response and the request that produced it go to
`$AUTORLBENCH_RESULTS/judge-raw.json`.
