# Recorded training session

The workspace contains a vision-language GRPO implementation, the recorded configuration, scalar history and validation outputs for one session. The recipe uses Qwen3-VL-8B-Instruct, image-bearing ScienceQA rows, seed42, four-way FSDP FULL_SHARD and DP2×SP2. Inspect the code and evidence to assess training correctness and explain the observed behavior.

Entry: `python scripts/run_training.py --config-only` composes the configuration. `python scripts/run_training.py` runs forty updates after resource preparation and four-device allocation. Recorded sampling uses validation n1 every five steps and a 200-step scheduler. Existing logs belong to the original session; a new run writes to runs/session.
