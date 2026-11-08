# Solution verifier

`test.sh` follows the same entry convention as
`configuration/verl_grpo_adamw_eps_too_large`;the task-specific grader is CPU-only.
It requires `report.md`,unchanged non-target code and source inputs,complete fresh
parquet output with the expected question/instruction prompts and labels,
and agreement with the actual launch configuration and
native dataset consumption. Every check is mandatory;score is exactly0 or1.

The grader does not execute submitted preprocessing inside the hidden verifier
sandbox. That execution happens in the earlier rerun sandbox,which has no oracle
mount. The verifier independently rechecks the actual rows and input capture,
rejects symlinks and missing files,and writes results atomically. No accuracy
threshold or preferred source diff is used.
