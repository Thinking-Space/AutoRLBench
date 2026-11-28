# Issue verifier

This verifier grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden `evaluations/issue/oracle.md` with an LLM judge. Verdict, root cause, and evidence are weighted 0.2 / 0.6 / 0.2, with the component conditions enforced by `judge.py`. The gold verdict is unhealthy; the essential cause is the FP32-to-BF16 destination copy in `checkpoint.py::load_hf_weights` before the router buffer is restored to FP32.

It requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`, and writes `result.json` plus `judge-raw.json` under `$AUTORLBENCH_RESULTS`.
