# Reference repair and evidence

## Reference implementation

`baseline/verl/workers/config/model.py` matches official verl v0.9.0, commit
`483b8a009ba3a97563edee3a19887e4862b8094a`. Relative to this source, the
workspace changes two imports and the processor factory in this module.
`source.diff` records the source difference.

The reference repair restores `hf_processor` in `HFModelConfig.__post_init__`
and its import. Processor initialization must preserve the multimodal RoPE
capabilities used by the agent loop. Alternative repairs are accepted when
the fresh position and log-probability checks pass and training completes.
Model, dataset, reward, and training budget remain fixed.

## Verification scope

The verifier imports the submitted workspace and exercises its model
configuration, agent-loop multimodal inputs and positions, engine input
preparation, forward step, and response log-probability extraction. The
measurement uses the pinned original checkpoint with PyTorch FSDP and no
optimizer. Its default world size is one (NO_SHARD);
`AUTORLBENCH_SCORE_GPUS=2` enables a two-process measurement. The training
experiment uses FSDP2 on eight GPUs.

Solution evaluation requires a fresh 20-step training run and fresh six-case
parity measurements. Every case must have correct engine positions and a
maximum absolute response-token log-probability difference below 1.0 nat.
Validation accuracy and reward have no independent pass threshold. The parity
measurement does not load the new step-20 checkpoint.

## Reference evidence

- `historical_evidence_manifest.json`: asset checksums, reference measurements,
  and training-completion summaries used by the verifier.
- `paired_measurements/`: identical captured inputs scored by the defective
  and reference implementations.
- `control_run/`: reference training logs and per-step metrics.
- `recorded_results/`: comparison tables and figures.
- `source_manifest.json`: workspace source checksums and reference overlay.

One run per side is available. The maximum frozen disagreement is
3.7402865886688232 nat for the defective implementation and
0.550462007522583 nat for the reference implementation. Relative to the
1.0 nat limit, the defective maximum is 2.7402865886688232 above the limit
and the reference maximum is 0.449537992477417 below it. Reference engine
positions match all six cases. Both training runs complete 20 steps with
finite nonzero gradients and repeated validation.

These measurements establish a position-semantics defect and persistent
training–inference inconsistency. They do not establish training collapse,
multi-seed variance, or that the final validation gap is entirely caused by
the repair. GPU measurements were obtained in the recorded Python runtime;
the shared Docker image still requires a build and fresh GPU validation.

The paired reference experiment is recorded here: `recorded_results/`
holds the frozen measurements, and the issue verifier's
`resources/qualification.json` stores the paired run used to calibrate
thresholds; checking those measurements requires no GPU, and replay does not
replace fresh solution evaluation.
