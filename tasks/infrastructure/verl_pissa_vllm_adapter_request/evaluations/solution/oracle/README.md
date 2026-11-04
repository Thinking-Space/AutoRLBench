# Canonical solution oracle

Repair the vLLM request path to test `VLLM_LORA_INT_ID` in the integer set
returned by `engine.list_loras()`. When present, attach the existing
`LoRARequest` so rollout generation and actor log-probability evaluation use
the same PiSSA policy.

The matched runs change only this membership test. At step 4 the defective run
has agreement `0.10476375009925555`; the repaired run has
`0.8907221660212565`. Grading requires `>= 0.80` at exactly step 4. The
defective arm's zero correctness reward is retained as authentic downstream
evidence and is not replaced by an auxiliary reward. One run per side supports
the margin but not a variance claim; step-zero health is unavailable.
