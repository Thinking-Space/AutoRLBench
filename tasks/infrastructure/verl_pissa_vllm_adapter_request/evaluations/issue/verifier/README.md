# Issue verifier

The LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden
oracle. Verdict, root cause and evidence are weighted 0.2 / 0.6 / 0.2; passing
requires at least 0.8. The essential attribution is the string-name versus
integer-ID mismatch that suppresses the vLLM `LoRARequest`.
