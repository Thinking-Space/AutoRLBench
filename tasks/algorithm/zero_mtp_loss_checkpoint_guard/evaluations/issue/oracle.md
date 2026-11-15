# Canonical issue oracle

Verdict: **unhealthy** for the **historical failure**.

The concrete fault is an incompatible unlabeled RL/MTP interface. The real Miles Megatron model wrapper requests `labels=None` and `position_ids=None` to return token logits for policy optimization. The model must retain MTP parameters for initialization/checkpoint/weight conversion, and `mtp_num_layers=1` with `mtp_loss_scaling_factor=0.0` does not by itself prevent Megatron from entering the auxiliary branch.

Historical causal chain: `GPTModel.forward:577` → `GPTModel._postprocess:647` → MTP block/layer forward at 1619/1291 → `_get_embeddings:913` → `roll_tensor:165` → `_roll_tensor_packed_seq:249`, which calls `.clone()` on missing position IDs. Training cannot complete an optimizer update at this boundary. This is not a policy-loss scalar plus a `None` auxiliary-loss value.

Evidence must cite the original failure under `workspace/logs/historical/`, the affected complete framework files, and the Miles nullable forward interface. The source exported by the source inspection matches these stack statements but is not proven byte-identical to the historical image; a mutable tag is insufficient proof.

The **corrected smoke** is a separate train-only replay on saved rollout data: eight real optimizer updates and checkpoint save. The preserved hook skips auxiliary postprocessing only for an unlabeled zero-scaled path, temporarily changes a copied config, and restores the original config in `finally`. Labeled or nonzero-auxiliary paths must be preserved; deleting MTP parameters is not equivalent.

New small random-model probes belong to a third evidence category. They can establish the same framework exception and a repaired executable path, real gradients/optimizer behavior, MTP parameter preservation, and a small checkpoint roundtrip. They do not establish a full 9B distributed RL rerun or held-out gains.

The checkpoint guard has exact source provenance and an independent debug-smoke exception. Do not require it as a second cause: no independent historical failure evidence establishes it caused historical failure. A diagnosis that calls this an unconditional optional-loss-addition TypeError, invents a coupled guard failure, blames an external reward service, or uses old synthetic 40-step telemetry as this failure's evidence is incorrect. Do not invent solution thresholds or a same-seed full-training comparison.
