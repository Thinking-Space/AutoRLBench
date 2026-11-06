# Missing difficult examples inflate training reward

Training JustRL-II-base-model with PPO on an UltraData-RL-2609 math selection with a fixed easy/hard mixture. At the midpoint the hard backing directory is relocated but the sample manifest still points at the old paths, so every hard example fails its file read and is silently resampled away — training reward rises on the surviving easy-only population while the fixed held-out evaluation does not move.

| | |
| --- | --- |
| **Author** | Pan Haoxuan |
| **Affiliation** | Shanghai Jiao Tong University |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `stale-manifest` `data-relocation` `survivorship-bias` `silent-failure` |
| **Framework** | JustRL-II (miles PPO), Megatron actor + critic + SGLang rollout + Ray |
| **Run time** | ~42 minutes |
| **Resources** | 3 GPU |

## Difficulty

The defect is in `workspace/loader.py::resolve`, the task's executable submission interface: it returns `record['path']`, the manifest's stale mapping. At the configured midpoint (step 4) the hard backing directory moves from `live/hard` to `archive/hard` without updating the manifest. `relocation_rollout.generate` reopens the selected file on each attempt, catches the `FileNotFoundError`, and emits an ABORTED sample; the native `check_no_aborted` filter rejects the group, and the rollout loop resamples until the 16-sample batch fills — with easy examples only.

Nothing fails loudly: training stays live, optimizer updates continue, checkpoints save, and training reward *rises* (survivor mean 0.4375 → 0.8125 at step 5). The single strongest piece of evidence: `data/survivor_reward_mean` (reward over admitted samples) climbs while `data/reward_per_read_attempt` — the same reward over all read attempts — stays flat at 0.31–0.41, because 16 of the 32 read attempts per step abort. Hard samples keep being attempted (`attempted_hard` 16 per step) while none are admitted after the midpoint, and the fixed held-out evaluation goes 0.375 → 0.3125 over the same window, against the training-reward rise.

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data/survivor_reward_mean` | 0.375 | 0.438 | 0.375 | 0.438 | 0.750 | 0.813 | 0.625 | 0.625 |
| `data/reward_per_read_attempt` | 0.375 | 0.438 | 0.375 | 0.438 | 0.375 | 0.406 | 0.313 | 0.313 |
| `data/admitted_hard` (of 16 attempted) | 8 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |

Unlike most tasks, the instruction does not announce a defect: the agent must first return a verdict (`healthy` / `unhealthy`) on the current implementation, and the observations under `logs/recorded/` come from an earlier run — the verdict must rest on the current workspace, not on old error logs. The fault is deliberately injected, not an unfixed upstream JustRL-II defect.

## Reference solution

- **issue**: a passing diagnosis returns `Verdict: unhealthy` and names both halves — the stale manifest still mapping hard sample IDs to `live/hard` after the relocation to `archive/hard`, and the recovery path that turns persistent missing data into a recoverable per-sample failure (caught `FileNotFoundError` → ABORTED → `check_no_aborted` group rejection → resampling until the batch fills with easy survivors) — connected to the survivor-biased reward. Attributions that must fail: the model's task being to learn a mapping, a reward bonus or fabricated reward, optimizer divergence as the primary event, SGLang scheduler cancellations counted as missing data, cached in-memory prompts disappearing when their source JSONL moves, and "training reward improved, therefore healthy".
- **solution**: keep the migration event and the original sample IDs, and make `resolve` consult the current storage state (the manifest updated to the new paths), so hard examples are admitted again. This preserves backing-file hashes, the intended easy/hard mixture, reward function, model and budget; a persistent-missing-data alarm or retry budget is also defensible, but hiding abort metrics or averaging over a different denominator does not restore training coverage. A healthy reference resolver is kept maintainer-side at `evaluations/solution/oracle/reference_loader.py`.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent submits `report.md` and the bounded `loader.py` — a restricted read-only path query, not arbitrary Python — and does not train; the harness restores the step-3 checkpoint and runs continuation steps 4–7 per `run_spec.json` (4 h timeout, 3 GPUs), then grades on the hidden starting verdict (`unhealthy`, `expected_verdict.json`), exact sample identities and content (8 easy + 8 hard per step), model/Adam/scheduler/RNG restoration, four real updates, final checkpoints and the fixed evaluation — reward improvement is explicitly not a pass requirement. Margins only, no variance: the calibration receipts under `provenance/receipts/` are the original GPU runs that predate the bounded query interpreter — one faulty injection recorded end to end, whose steps 0–3 serve as the non-migrated control — and no new GPU calibration runs are claimed. Operational: the sealed scene is 66.7 GB, pinned locally with no public checkpoint download, and requires the Docker backend with separate prepare / rerun / verifier containers.
