# Canonical diagnosis — qwen3vl_geo3k

Expected verdict: **unhealthy**.

Grade the diagnosis against the implementation, its effect on multimodal
positions, and the training–inference evidence described below.

## Root cause and causal chain

The target is `workspace/verl/verl/workers/config/model.py`,
`HFModelConfig.__post_init__`. It initializes `self.processor` using raw
`AutoProcessor.from_pretrained` rather than verl's `hf_processor` adapter.

The two factories can return the same nominal `Qwen3VLProcessor`, tokenizer,
image processor, image grids, and prompt IDs. However, the verl adapter in
`verl/utils/tokenizer/tokenizer.py` also attaches `config`, binds
`Qwen3VLModel.get_rope_index` to the processor, and binds auxiliary vision
position helpers when required by Transformers. Those capabilities are lost.

`AgentLoopWorker` reads `self.model_config.processor`. Its existing
`_compute_position_ids` guard treats a processor without `get_rope_index` as
eligible for standard one-dimensional positions. Consequently, image batches
reach the FSDP trainer with flat `cumsum(mask)-1` coordinates. The model accepts
these legal-shaped coordinates and does not reconstruct the missing visual
coordinates. vLLM computes its positions independently. Equal weights and token
IDs are therefore scored using different positional semantics.

The defect is the loss of a required processor capability. A mismatch in the
nominal processor class name is not required to identify it.

## Required evidence

The snapshot contains a 20-step GRPO run under `workspace/logs`, effective
configuration, offline W&B text files, held-out validation scores, and
representative generated outputs. The paired reference measurements establish:

- Identical frozen model, images, prompt IDs, response IDs, and image grids.
- Wrong visual positions at the actual engine input; correct text control.
- Severe candidate image-token-context logprob mismatch, with finite values.
- Clean upstream control passes the numerical baseline and position checks.
- Restoring the factory removes the large discrepancy without changing the
  model, data, reward, sampling distribution, or causal-label alignment.
- Both candidate and healthy GRPO runs complete with finite nonzero gradients.

The largest frozen response-token log-probability difference is 3.740287 nat
in the defective workspace and 0.550462 nat with the reference initialization.
All five visual cases have incorrect positions in the defective workspace;
the reference positions match in all six cases. During training, the mean
`rollout_corr/kl` is 0.006591 versus 0.000623 nat/token. These are distinct
metrics: the frozen maximum is not a training-step average or a measurement
of the final checkpoint. There is one reference run per side.

## Attribution boundaries

Normal-looking loss, finite gradients and a completed job do not establish
correct positional semantics. A generic assertion that "BF16 kernels differ"
does not identify this root cause. Neither does blaming the geometry3k reward,
GRPO algorithm, an arbitrary hyperparameter, or an off-by-one causal label shift.
The upstream engine's `torch.roll` is intact in this task.

Do not require a particular spelling of the diagnosis or source-level repair.
A correct answer must connect the initialization loss of capability to the
actual flat-position fallback and trainer/rollout mismatch. Do not require
reward collapse or a held-out accuracy improvement: those have not been
established as consequences of this task's defect.
