# Unlabeled RL forward enters the auxiliary MTP path

Training Qwen3.5-9B with GRPO on a mixed code-repair corpus fails before its first optimizer update because an unlabeled RL forward enters Megatron's retained MTP branch and rolls absent position IDs despite zero auxiliary loss scaling.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Algorithm |
| **Domain** | Code RL |
| **Tags** | `mtp` `zero-loss` `nullable-input` `rl-training` |
| **Framework** | Miles (GRPO; `0.2.1.dev1609+g48f39cb4`), Megatron-Core, SGLang |
| **Run time** | ~40 minutes (recorded master-log span) |
| **Resources** | 16 × H100 (80 GB) |

## Difficulty

Miles requests policy logits with `labels=None` and `position_ids=None`. The model retains `mtp_num_layers=1` for parameter and checkpoint compatibility while setting `mtp_loss_scaling_factor=0.0`. The coefficient does not disable execution: `GPTModel._postprocess` still invokes MTP, whose `_get_embeddings` rolls the absent position IDs; `_roll_tensor_packed_seq` then calls `None.clone`.

The historical failure completes 0 optimizer updates at this boundary. Its continuous evidence is under `workspace/logs/historical/`; the separate corrected smoke in `evaluations/issue/historical_control/` replays saved rollout data on 8 actor GPUs, completes 8 updates, and saves a checkpoint. A diagnosis must connect the actual nullable interface to this stack rather than blame optional-loss addition, model quality, or an external service. The checkpoint guard is an ancillary source artifact, not a second established cause of the historical failure. The original master/worker API archives retain 5,694/275 record positions, and the separate corrected archive contains 2,228 master records plus all 8 step metrics. These are complete API exports for the selected training pods with private payloads explicitly redacted; they do not include every external service file.

The workspace includes generic framework source with unrelated, unverified optional Miles extensions omitted and a bounded random tiny-model probe with a real MTP1 block, forward, backward, AdamW, and a small checkpoint roundtrip. The default new probe budgets 2 updates on 1 GPU and is distinct from both historical runs. The source inspection matches the historical stack statements, but does not establish byte-identical historical image provenance. The entrypoint prioritizes the vendored source over the environment's native Core; the packaged probe has been validated on 1 GPU in the platform mirror of the public Slime image.

Issue-only: the historical corrected smoke establishes a functional repair, but supplies no calibrated outcome margin for grading arbitrary repairs.

## Reference solution

- **issue**: identify the unlabeled RL/MTP interface mismatch and the `_postprocess → _get_embeddings → roll_tensor → _roll_tensor_packed_seq → None.clone` chain. Explain why zero auxiliary scaling alone does not skip MTP and why removing MTP parameters would change checkpoint compatibility. The preserved repair bypasses auxiliary postprocessing only for unlabeled zero-scaled logits and restores the original config in `finally`; it preserves labeled and nonzero-auxiliary paths.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2; passing requires verdict = 1.0, root cause > 0.8, and evidence > 0.8. The oracle distinguishes the historical failure, corrected smoke, source inspection, and new reduced framework probe. New probe results are execution checks, not held-out gains or a replacement for historical evidence.

In the separate platform-mirror validation, the default packaged rerun fails as expected during `real_model_forward` with `AttributeError` at `None.clone` and 0 optimizer updates. With the reference hook supplied explicitly, the same reduced probe exits successfully after 2 real backward/AdamW updates and a model/optimizer checkpoint roundtrip. Its report confirms `base_parameters_updated=true`, `mtp_weights_preserved=true`, and `config_restored=true`. This validates the packaged failure and corrected execution path; it does not constitute a full Qwen3.5-9B RL rerun or a calibrated solution-level score.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The final reduced framework failure/reference validation uses the resulting source; report source identities match the publication snapshot.
