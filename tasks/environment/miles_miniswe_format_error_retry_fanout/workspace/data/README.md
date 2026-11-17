# Selected SWE-bench Verified data

`prepare_data.py` selects 16 complete records from the
`princeton-nlp/SWE-bench_Verified` test split. Selection is deterministic:
group records by repository in dataset order, then choose one record from each
repository in round-robin order until 16 records are selected.

The selection excludes instance IDs documented by the public Harbor SWE-bench
adapter as known environment or verifier outliers.

Generated files:

- `swebench_verified.jsonl`: complete dataset records used to build Harbor tasks.
- `miles_prompts.jsonl`: Miles prompt rows with `instance_id` and
  `agent_name=mini-swe-agent`.
- `selected_instance_ids.txt`: the 16 instance IDs, in selection order.

The RL run samples each of the 16 prompts four times, producing 64 logical
rollouts for one GRPO update.
