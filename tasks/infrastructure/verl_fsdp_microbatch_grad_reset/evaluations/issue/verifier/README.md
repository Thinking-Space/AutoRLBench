# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`evaluations/issue/oracle.md` with an LLM judge (`judge.py` +
`judge_prompt.md`).

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition. The gold
verdict for this task is `unhealthy`; the essential root cause is the `self.module.zero_grad(set_to_none=True)` added inside the micro-batch loop of `verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.forward_backward_batch`, which discards the gradients of all but the last micro-batch, so each optimizer step uses 32 of the mini-batch's 256 samples.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
The output is written to `$AUTORLBENCH_RESULTS/result.json`, and the raw judge response plus the
request that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.
