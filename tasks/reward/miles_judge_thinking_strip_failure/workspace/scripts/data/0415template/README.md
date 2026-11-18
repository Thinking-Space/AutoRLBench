# 0415template SFT data — mix subset

Build the **megatron 4k ∪ repeat5033 4k** SFT mix and (optionally) convert to sstable.

## Quick run

```bash
# 1) build mix jsonl (6264 unique rows, ~381MB)
python3 scripts/data/0415template/prepare_mix.py

# 2) convert to sstable for SFT (auto-installs deps; needs internet)
bash scripts/data/0415template/convert_to_sstable.sh
```

## What's here

| File | Purpose |
|------|---------|
| `prepare_mix.py` | Read megatron 4k + repeat5033 4k from `/user/lizhen/data/autocuda_sft/0415template/raw/`, union dedup by question SHA1, write merged jsonl |
| `convert_to_sstable.sh` | Renames `{question,response}` → `{input,output}`, installs deps, runs `zyx_infra/trans_sstable.py` with qwen3 tokenizer |
| `README.md` | This file |

## Key facts about the mix

- megatron 4k and repeat5033 4k share **1736 questions** (43% overlap)
- For those 1736 overlapping questions, **responses are byte-identical** (verified, 0 divergent)
- Union dedup yields **6264 unique** rows = 4000 + (4000 - 1736)
- Megatron-first stable ordering → reproducible

## Output paths (under `/user/luozhenghang/0226_miles/data/sft/0415template/`)

| Path | Format |
|------|--------|
| `raw/sft_data_mix4k_union.sample_6264.jsonl` | jsonl, schema `{question, response}` |
| `sft_data_mix4k_union.qwen3_8b.0415template/sft_data_mix4k_union.qwen3_8b.0415template.sstable` | sstable, after `convert_to_sstable.sh` |

## When to use mix vs single

| Goal | Pick |
|------|------|
| Reproduce lizhen baseline | `sft_data_repeat5033.sample_4000.qwen3_8b.0415template/` |
| Diversify SFT init for Qwen3-8B | mix (this) |
