"""Prepare pinned JustRL inputs for the native PPO launch."""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# Match the harness fetcher's HTTPS path when the Hugging Face mirror blocks Xet.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.modules.setdefault("hf_xet", None)

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MODEL_FILES = {
    "LICENSE": "1c8f573e830ca9b3ebfeb7ace1823146e22b66f99ee223840e7637c9e745e1c7",
    "config.json": "37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41",
    "generation_config.json": "72cbec1015da9ed03ad025483005cbf6481403abf947adfd54e504d1a66a2126",
    "model.safetensors": "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
    "tokenizer.json": "88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773",
    "tokenizer_config.json": "8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5",
}
DATA = "BytedTsinghua-SIA/DAPO-Math-17k"
DATA_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DATA_DIGEST = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
JUSTRL_REVISION = "cf444e9920b599c865ee6742c88dc162dcabcec2"
VAL_ASSETS = {
    "AIME24": (12495, "147a9f9031b37e9b12450dfd4f2050a1bece8b3550bb6b9907fa386b5fedf887"),
    "AIME25": (15934, "9d2333e1dc8831284747dfa62b58d8b6a84a94090ee108d1dd101f8d20a38593"),
    "AMC23": (11770, "01191828e1fe738a2d19c3379213a6e5c92b425e1f86d67323e7b8613efa5678"),
}
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def check_sha(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Asset SHA-256 mismatch: {path.name}")


def messages(row):
    return [{"role": "user", "content": row["prompt"][0]["content"] + SUFFIX}]


def example_id(row):
    return row["prompt"][0]["content"], str(row["reward_model"]["ground_truth"])


def write_train(source, output, tokenizer):
    prompts = {}
    for batch in source.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            prompts.setdefault(example_id(row), messages(row))

    allowed = set()
    keys = list(prompts)
    for start in range(0, len(keys), 256):
        batch = keys[start:start + 256]
        tokens = tokenizer.apply_chat_template(
            [prompts[key] for key in batch], add_generation_prompt=True,
            tokenize=True, return_dict=False,
        )
        allowed.update(key for key, ids in zip(batch, tokens, strict=True) if len(ids) <= 1024)

    if (len(prompts), len(allowed)) != (17405, 17395):
        raise ValueError("Pinned training inputs no longer match the recorded prompt filter")

    with pq.ParquetWriter(output, source.schema_arrow, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=8192):
            rows = [row for row in batch.to_pylist() if example_id(row) in allowed]
            for row in rows:
                row["prompt"] = messages(row)
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=source.schema_arrow))
    if pq.read_metadata(output).num_rows != 1790700:
        raise ValueError("Unexpected training row count")


def main():
    assets = Path(os.environ["AUTORLBENCH_WORKSPACE"]) / "assets"
    model = assets / "model"
    data = assets / "data"
    data.mkdir(parents=True, exist_ok=True)
    model_snapshot = Path(snapshot_download(
        MODEL, revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
        cache_dir=Path(os.environ["AUTORLBENCH_MODELS"]) / "huggingface",
    ))
    model.mkdir(parents=True, exist_ok=True)
    for name, digest in MODEL_FILES.items():
        source = model_snapshot / name
        check_sha(source, digest)
        destination = model / name
        if not destination.exists():
            try:
                os.link(source.resolve(strict=True), destination)
            except OSError:
                shutil.copyfile(source, destination)
                destination.chmod(0o644)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cached = Path(hf_hub_download(
        DATA, "data/dapo-math-17k.parquet", repo_type="dataset",
        revision=DATA_REVISION, cache_dir=Path(os.environ["AUTORLBENCH_DATASETS"]) / "huggingface",
    ))
    check_sha(cached, DATA_DIGEST)
    write_train(pq.ParquetFile(cached), data / "train.parquet", tokenizer)

    for name, (size, sha256) in VAL_ASSETS.items():
        url = f"https://raw.githubusercontent.com/thunlp/JustRL/{JUSTRL_REVISION}/data/{name}/test.parquet"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read(size + 1)
        if len(body) != size or hashlib.sha256(body).hexdigest() != sha256:
            raise ValueError(f"Validation source identity mismatch: {name}")
        rows = pq.read_table(pa.BufferReader(body)).to_pylist()
        for row in rows:
            row["prompt"] = messages(row)
        tokens = tokenizer.apply_chat_template(
            [row["prompt"] for row in rows], add_generation_prompt=True, tokenize=True, return_dict=False,
        )
        if any(len(ids) > 1024 for ids in tokens):
            raise ValueError(f"Validation prompt exceeds recipe length: {name}")
        pq.write_table(pa.Table.from_pylist(rows), data / f"{name}.parquet", compression="zstd")
    print(json.dumps({"train_rows": 1790700, "validation_rows": {k: pq.read_metadata(data / (k + ".parquet")).num_rows for k in VAL_ASSETS}}))


if __name__ == "__main__":
    main()
