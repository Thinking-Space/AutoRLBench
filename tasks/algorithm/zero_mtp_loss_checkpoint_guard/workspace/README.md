# Historical MTP failure and a real framework probe

Start with `logs/historical/manifest.json` and the continuous historical logs. It covers the original training attempt; inspect its manifest for archive coverage and redaction rules.

`vendor/megatron/` contains the complete affected generic Megatron package exported by the source inspection. Its relevant stack statements match the old failure, but the image tag may have drifted. `vendor/miles/` comes from the historical pinned Miles wheel. Source identities belong to `../environment/source_manifest.json`; use the bundled source rather than a host checkout.

The local probe creates a tiny random real `GPTModel` with a real one-layer MTP block, packed sequences, and `labels=None/position_ids=None`. It uses CUDA/TransformerEngine, real backward and AdamW, with 2 steps by default and at most 8. No trained weights, tokenizer, task dataset, model service, or reward service is loaded. Its synthetic policy-gradient surrogate is only an execution probe, not a new RL result.

```bash
timeout 600s python3 -B scripts/train_reproduction.py --steps 2
```

The faulty source is expected to raise before any optimizer update. Fix validation additionally checks base-parameter updates, MTP config/key/weight preservation and a real small model/optimizer checkpoint roundtrip. This checkpoint is a PyTorch smoke artifact, not the historical distributed checkpoint.

`--guard-only` executes the preserved Bash checkpoint helper (identifier-only replacements recorded) without importing CUDA. The guard is an ancillary schedule check, not a second established cause of the historical crash. The full probe records its decision but bypasses it to expose the model fault. `--hook-file` and `--guard-file` are optional source-selection arguments.

Historical evidence is immutable. New outputs are written only under the selected `--output-dir` (default `experiment_logs/`).

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `../environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The final reduced framework failure/reference validation uses the resulting source; report source identities match the publication snapshot.
