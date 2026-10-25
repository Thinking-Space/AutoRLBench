# Issue verifier

`test.sh` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`
with an LLM judge. It does not read `submission.json`.

The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`. The gold verdict
for this task is `unhealthy`; the root cause is the lost GPU→CPU transfer in
`Encoder.encode` (`Search-R1/search_r1/search/retrieval_server.py:117-120`) that
leaves every FAISS query vector all-zero.

Two misreadings the oracle calls out explicitly, because this task invites them:

- "the run does not learn" — it does, by 12.5 points of EM. The defect is the
  shortfall against the recorded control (32.8% EM), not an absence of learning.
- "the FAISS index is on CPU while the encoder is on GPU" — true but not the
  defect by itself; it only counts as an attribution if it names the unwritten
  `response_buffer` as the reason the data never crosses.

`judge.py`, `judge_prompt.md` and `test.sh` are byte-identical to
`tasks/slime_search_r1_nomask/evaluations/issue/verifier/`; only `oracle.md`,
`gold.yaml` and this file are task-specific.

`gold.yaml` is retained as a reviewer-facing reference. Machine grading reads
`oracle.md`, not `gold.yaml`.

Requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
