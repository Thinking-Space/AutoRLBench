# Pinned public runtime helpers

These files are copied without modification from the existing public JustRL2
AutoRLBench proposal at commit `73090dd29837afe7dc50b59863051bf2785f0562`,
`merak0514/AutoRLBench`, path `proposals/miles_ppo_empty_shard_collective/justrl2/`.
They retain their Apache-2.0 SPDX headers. No earlier CP defect is injected here.

- `runtime_evidence.py`: actual optimizer/master-parameter observations.
- `process_group_compat.py`, `sglang_compat.py`: image/parser compatibility.
- `checkpoint_evidence.py`: selected native checkpoint inventory and hashes.
- `source_files.json`: inventory of public JustRL-II revision
  `75814318f243af7e39af337c1173fc3927c9635b`.

The source fixtures in `../fixtures/` are unmodified public JustRL-II files at
that same revision, used only to test adapter method contracts without importing
GPU libraries. Source archive and all source files are verified again on cluster.

These helpers contain neither private dataset rows nor model weights. See the
parent experiment notes for runtime limitations and actual validation status.
