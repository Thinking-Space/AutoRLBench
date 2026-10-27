# Hindsight-skill SFT scripts

Every dataset exposes the same two entrypoints:

```bash
bash scripts/sft/<dataset>/prepare_data.sh
bash scripts/sft/<dataset>/train_sft.sh
```

Supported datasets are `alfworld`, `search`, `webshop`, `ezpoints`, and
`sokoban`. `prepare_data.sh` collects baseline rollouts, generates one
episode-level skill per trajectory, and writes the train/validation parquet
files. `train_sft.sh` trains for three epochs by default and exports the latest
Hugging Face checkpoint under `$MODELS_ROOT`.

The public training scripts use one shared implementation in
`scripts/sft/_common/trainer.sh`; each dataset wrapper contains only its
model, learning rate, context length, environment, and multimodal settings.
The other files under `_common` provide teacher-name inference and Python
helpers used by the text pipelines. They are internal files rather than
additional launchers.

Default training settings:

| Dataset | Base model | Learning rate | Max length | Multimodal |
| --- | --- | ---: | ---: | --- |
| ALFWorld | Qwen2.5-3B-Instruct | 5e-6 | 8192 | no |
| Search QA | Qwen2.5-3B-Instruct | 5e-6 | 12288 | no |
| WebShop | Qwen2.5-3B-Instruct | 5e-6 | 12288 | no |
| EZPoints | Qwen2.5-VL-3B-Instruct | 2e-6 | 4096 | yes |
| Sokoban | Qwen2.5-VL-3B-Instruct | 2e-6 | 4096 | yes |

All local policy vLLM processes use `--gpu-memory-utilization 0.6`. Common
overrides include `MODEL_PATH`, `DATA_DIR`, `NPROC_PER_NODE`,
`TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE_PER_GPU`,
`MAX_LENGTH`, and `EXPORT_MODEL_DIR`.

The data directory and exported model name include the inferred teacher suffix,
for example `glm_self` for data and `glm-self` for the model. Set
`SFT_TEACHER_SHORT` explicitly if the configured teacher model name is not
recognized.
