# Issue verifier

The LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden
oracle, scoring `verdict_score`, `root_cause_score` and `evidence_score` from
0 to 1. Passing requires `verdict_score == 1.0` with both `root_cause_score`
and `evidence_score` above `0.8`; the 0.2 / 0.6 / 0.2 weighted sum is
reported as the score but is not the pass rule. The judge prompt scores each
oracle root cause and evidence item as a boolean and reports the fractions.
The essential attribution is that disabling a PiSSA adapter exposes the
residual checkpoint rather than the pretrained base.
