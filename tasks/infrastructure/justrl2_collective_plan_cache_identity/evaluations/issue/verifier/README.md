# Issue verifier

Grades `$AUTORLBENCH_RESULTS/agent_answer.md` against `evaluations/issue/oracle.md` using `judge.py` and `judge_prompt.md`.

Weights are 0.2 (verdict) / 0.6 (root cause) / 0.2 (evidence); passing requires `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`. Gold verdict is `unhealthy`; the essential root cause is `miles/utils/collective_plan.py::_signature` omitting communicator `group` identity from its `_plans` cache key.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`. Writes `$AUTORLBENCH_RESULTS/result.json` and `$AUTORLBENCH_RESULTS/judge-raw.json`.
