---
name: convert-checkpoints
description: Convert Miles model checkpoints between HuggingFace and Megatron torch_dist formats. Use when the user asks to 转参数, convert HF to torch_dist, convert torch_dist to HuggingFace, export checkpoints, or prepare MiniCPM5 checkpoints.
---

# Convert Checkpoints

## Scope

Use this skill for Miles checkpoint conversion:

- HuggingFace format -> Megatron `torch_dist`
- Megatron `torch_dist` -> HuggingFace format
- MiniCPM5 conversions using scripts under `scripts/minicpm5`
- Other model conversions by adapting `tools/convert_hf_to_torch_dist.py` and `tools/convert_torch_dist_to_hf.py`

## Missing Inputs

If the user request does not include enough information, ask for the missing inputs before running or writing conversion commands. Required inputs:

- Conversion direction: HF -> `torch_dist` or `torch_dist` -> HF.
- Model family/size: for example MiniCPM5 `1b`, `2_6b`, `16a3b`, or another model.
- Source path: HF checkpoint path or `torch_dist` iteration directory.
- Output path: target `torch_dist` save path or HF output directory.
- For `torch_dist` -> HF: original HF model directory for tokenizer/config assets, unless `--model-name` is intentionally used.
- For non-MiniCPM5 models: the model args source, such as a script under `scripts/models` or an existing launch script.

Only infer values from the repository when they are explicitly present in the referenced script or file. Do not silently use hard-coded example paths from scripts as the user's intended paths.

## First Checks

1. Identify model family and size: MiniCPM5 `1b`, `2_6b`, `16a3b`, or another model.
2. Identify direction: HF -> `torch_dist` or `torch_dist` -> HF.
3. Confirm source path, output path, and model architecture args.
4. For `torch_dist` input, check the target directory is an iteration directory such as `iter_0000149`, containing `common.pt` and `.metadata`.
5. For HF output, keep `ORIGIN_HF_DIR` pointed at the original HF model so tokenizer/config/modeling files are copied.

## MiniCPM5 HF -> torch_dist

Prefer the model-specific scripts:

```bash
HF_CHECKPOINT=/path/to/hf_model SAVE=/path/to/torch_dist_ckpt \
bash scripts/minicpm5/16a3b/convert_minicpm5_16a3b.sh
```

Available MiniCPM5 examples:

- `scripts/minicpm5/1b/convert_minicpm5_1b.sh`
- `scripts/minicpm5/2_6b/convert_minicpm5_26b.sh`
- `scripts/minicpm5/16a3b/convert_minicpm5_16a3b.sh`

These scripts source the matching model args file, set `PYTHONPATH`, and call:

```bash
torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${SAVE}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --bf16 \
  "${MODEL_ARGS[@]}"
```

For larger models, only increase `--nproc-per-node` if the conversion script and model parallel args are consistent. Keep `WORLD_SIZE <= num_layers`.

## MiniCPM5 torch_dist -> HF

Prefer the model-specific scripts:

```bash
ORIGIN_HF_DIR=/path/to/original_hf \
bash scripts/minicpm5/16a3b/convert_minicpm5_16a3b_torch_dist_ckpt_to_hf.sh \
  /path/to/torch_dist/iter_0000149 \
  /path/to/output_hf
```

Available MiniCPM5 examples:

- `scripts/minicpm5/1b/convert_minicpm5_1b_torch_dist_ckpt_to_hf.sh`
- `scripts/minicpm5/2_6b/convert_minicpm5_26b_torch_dist_ckpt_to_hf.sh`
- `scripts/minicpm5/16a3b/convert_minicpm5_16a3b_torch_dist_ckpt_to_hf.sh`

These scripts extract `--vocab-size` from the model args and call:

```bash
python3 tools/convert_torch_dist_to_hf.py \
  --input-dir "${INPUT_DIR}" \
  --origin-hf-dir "${ORIGIN_HF_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --vocab-size "${VOCAB_SIZE}" \
  --force
```

## Other Models

For non-MiniCPM5 models:

1. Find the closest model args script in `scripts/models` or the training launch script.
2. Source that args file before calling `tools/convert_hf_to_torch_dist.py`.
3. Keep `--hf-checkpoint`, `--save`, TP/PP sizes, precision, vocab size, and architecture args aligned with training.
4. For `torch_dist` -> HF, pass either `--origin-hf-dir` or `--model-name`; prefer `--origin-hf-dir` to preserve tokenizer and config assets.
5. If tensor names fail to map, inspect `miles/backends/megatron_utils/megatron_to_hf` and model-specific bridge code in `miles_plugins`.

## Validation

After conversion:

- HF -> `torch_dist`: verify checkpoint tracker and iteration directory exist under the output.
- `torch_dist` -> HF: verify `config.json`, tokenizer files, `model.safetensors.index.json`, and safetensor shards exist.
- Run a lightweight load test when possible: load tokenizer/config for HF, or start a short dry-run/eval for `torch_dist`.
- Do not overwrite existing outputs unless the user explicitly accepts `--force` or a clean output path.
