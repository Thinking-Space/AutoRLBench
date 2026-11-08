# Qwen2.5 / HotpotQA

This is a historical Search-R1 GRPO experiment on the HotpotQA distractor
subset. Its goal is to learn evidence retrieval and reasoning from source
passages. Training and validation labels provide supervised scoring targets.

`config.json` records the launch settings. `catalog.py` records the catalog
materialization logic. `serve.py` exposes the configured catalogs to the agent
loop; `validation.py` selects the validation endpoint. Model-generated tokens
are scored separately from returned observations.

The staged inputs are under `shared/data/`. The model files materialized by
prepare are under `shared/models/actor/` and `shared/models/encoder/`.
`python3 run.py --check` validates imports and assets without a GPU.
The historical training configuration uses one GPU. The issue evaluation
uses the frozen snapshot and does not require training.

`logs/recorded/` contains the training console, all 32 per-update scalar files,
training trajectories for updates 1, 16 and 32, primary validation records at
updates 0, 16 and 32, and retrieval requests from the recorded trajectory.
Paths and identifiers have been normalized. Query text, document contents,
model responses, rewards, scores and retained numeric metrics are unchanged.
