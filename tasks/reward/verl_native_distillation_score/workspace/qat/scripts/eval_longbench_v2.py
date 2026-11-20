#!/usr/bin/env python3
"""Evaluate a causal LM on a LongBench v2 length/difficulty subset.

This evaluator is deliberately separate from ``eval_aime.py``.  LongRLVR is
the training data for the Qwen2.5-7B-1M line; LongBench v2 is its test
benchmark.  The Hugging Face dataset calls its only split ``train``.  That
name is a dataset-layout detail, not a request to use the training data.

The current experiment contract is ``--subset medium --difficulty all`` only, no prompt
truncation, Thinking disabled, and at most 4096 generated tokens.  The prompt
is the official LongBench v2 multiple-choice format with a final ``The correct
answer is (A/B/C/D)`` instruction so scoring is deterministic and auditable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


DATASET_ID = "THUDM/LongBench-v2"
DATASET_SPLIT = "train"  # HF's only split; it is the benchmark/eval split.
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_SUBSET = "medium"
DEFAULT_DIFFICULTY = "all"
DEFAULT_EXPECTED_ROWS = 215  # all LongBench v2 rows with length=medium.
DEFAULT_INSTRUCTION_SUFFIX = (
    "\n\nLet's think step by step and end your response with exactly one line "
    "of the form: The correct answer is (A), (B), (C), or (D)."
)
REQUIRED_FIELDS = (
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
)


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _source_sha256() -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with Path(__file__).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _load_rows_from_path(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("LongBench v2 parquet input requires pyarrow") from exc
        rows = pq.read_table(path).to_pylist()
    else:
        text = path.read_text(encoding="utf-8")
        if suffix in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else payload.get("data", payload)
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"LongBench v2 input must contain a list of objects: {path}")
    return [dict(row) for row in rows]


def _load_rows_from_hf() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("LongBench v2 evaluation requires datasets or a local data.json") from exc
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    return [dict(row) for row in dataset]


def load_longbench_records(
    path: Path | None = None,
    *,
    subset: str = DEFAULT_SUBSET,
    difficulty: str = DEFAULT_DIFFICULTY,
    limit: int | None = None,
    expected_rows: int | None = DEFAULT_EXPECTED_ROWS,
) -> List[Dict[str, Any]]:
    """Load and filter the benchmark, never truncating a prompt."""

    if subset not in {"short", "medium", "long", "all"}:
        raise ValueError("subset must be short, medium, long, or all")
    if difficulty not in {"easy", "hard", "all"}:
        raise ValueError("difficulty must be easy, hard, or all")
    if path is None:
        rows = _load_rows_from_hf()
    else:
        # A caller that names a local file must get that exact file.  Do not
        # silently replace a missing path with a different HF revision.
        rows = _load_rows_from_path(path)
    selected = []
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"LongBench v2 row {row.get('_id')} is missing {missing}")
        if subset != "all" and str(row["length"]).lower() != subset:
            continue
        if difficulty != "all" and str(row["difficulty"]).lower() != difficulty:
            continue
        selected.append(row)
    # A bounded smoke run is allowed to request ``--limit 1``.  Full runs
    # (no limit, or a limit at least the expected size) must match the pinned
    # governed subset cardinality.
    if expected_rows is not None and (limit is None or limit >= expected_rows) and len(selected) != expected_rows:
        raise ValueError(
            f"LongBench v2 subset={subset}, difficulty={difficulty} has {len(selected)} rows, expected {expected_rows}; "
            "check the dataset revision before evaluating"
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def build_prompt(row: Mapping[str, Any], *, thinking: bool = False) -> str:
    """Build the official LongBench v2 multiple-choice prompt."""

    prompt = (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        f"{str(row['context']).strip()}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {str(row['question']).strip()}\n"
        "Choices:\n"
        f"(A) {str(row['choice_A']).strip()}\n"
        f"(B) {str(row['choice_B']).strip()}\n"
        f"(C) {str(row['choice_C']).strip()}\n"
        f"(D) {str(row['choice_D']).strip()}"
    )
    if thinking:
        prompt += DEFAULT_INSTRUCTION_SUFFIX
    else:
        prompt += "\n\nEnd your response with exactly one line of the form: The correct answer is (A), (B), (C), or (D)."
    return prompt


def extract_answer(response: str) -> Optional[str]:
    """Use the LongBench v2 answer line, then the explicit final-answer form."""

    if not response:
        return None
    cleaned = response.replace("*", "")
    matches = re.findall(r"The correct answer is\s*\(?([A-D])\)?", cleaned, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"(?:final answer|answer)\s*(?:is|=|:)\s*\(?([A-D])\)?", cleaned, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if name == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 evaluation requires a CUDA device with bf16 support")
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _resolved_dtype(dtype: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, str(dtype).replace("torch.", ""))


def _load_model(args: argparse.Namespace, device: torch.device):
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("LongBench v2 evaluation requires transformers") from exc
    match = re.match(r"^(\d+)\.(\d+)", transformers.__version__)
    if match is None or tuple(int(part) for part in match.groups()) < (4, 37):
        raise RuntimeError(f"Transformers {transformers.__version__} is too old for Qwen2")
    dtype = _dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("LongBench evaluation requires a tokenizer with pad_token_id")
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        model = model.to(device)
    model.eval()
    hook = None
    if args.kivi:
        from qat_kv.kivi_qwen2 import install_kivi_hf_rollout

        hook = install_kivi_hf_rollout(
            model,
            bits=2,
            group_size=args.kivi_group_size,
            residual_length=args.kivi_residual_length,
            training_fake_quant=False,
            require_kernel=True,
        )
    elif args.kv_bits < 32:
        from qat_kv import attach_kv_fake_quant_hooks

        hook = attach_kv_fake_quant_hooks(
            model,
            bits=args.kv_bits,
            preserve_projection_bias=args.preserve_projection_bias,
            exact_bias_free_recompute=args.exact_bias_free_recompute,
        )
    return model, tokenizer, hook, _resolved_dtype(dtype)


def _encode(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return ids.to(device)


@torch.inference_mode()
def _generate(model: Any, tokenizer: Any, prompt: str, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    input_ids = _encode(tokenizer, prompt, device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=max(0, args.top_k),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **({"cache_implementation": "dynamic"} if args.kivi else {}),
    )
    completion = output[0, input_ids.shape[-1] :]
    eos_seen = bool(
        tokenizer.eos_token_id is not None
        and completion.numel()
        and torch.eq(completion, int(tokenizer.eos_token_id)).any().item()
    )
    generated_tokens = int(completion.numel())
    hit_max = generated_tokens >= args.max_new_tokens
    return {
        "response": tokenizer.decode(completion, skip_special_tokens=True),
        "prompt_tokens": int(input_ids.shape[-1]),
        "prompt_truncated": False,
        "response_tokens": generated_tokens,
        "eos_seen": eos_seen,
        "hit_max_tokens": hit_max,
        "truncated": bool(hit_max and not eos_seen),
        "finish_reason": "eos" if eos_seen else ("length" if hit_max else "stopped"),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    data_path = args.data
    rows = load_longbench_records(
        data_path,
        subset=args.subset,
        difficulty=args.difficulty,
        limit=args.limit,
        expected_rows=args.expected_rows,
    )
    source_count = len(rows)
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    indexed_rows = list(enumerate(rows))
    if args.num_shards > 1:
        indexed_rows = indexed_rows[args.shard_index :: args.num_shards]
    rows = [row for _, row in indexed_rows]
    model, tokenizer, hook, dtype_resolved = _load_model(args, device)
    predictions: List[Dict[str, Any]] = []
    correct = 0
    response_counts: List[int] = []
    prompt_counts: List[int] = []
    truncated = 0
    eos_samples = 0
    extraction_failures = 0
    started = time.time()
    try:
        for local_index, (index, row) in enumerate(indexed_rows):
            prompt = build_prompt(row, thinking=args.thinking)
            generation = _generate(model, tokenizer, prompt, args, device)
            prediction = extract_answer(generation["response"])
            expected = str(row["answer"]).strip().upper()
            is_correct = prediction == expected
            correct += int(is_correct)
            response_counts.append(int(generation["response_tokens"]))
            prompt_counts.append(int(generation["prompt_tokens"]))
            truncated += int(generation["truncated"])
            eos_samples += int(generation["eos_seen"])
            extraction_failures += int(prediction is None)
            predictions.append(
                {
                    "index": index,
                    "id": row["_id"],
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                    "length": row["length"],
                    "answer": expected,
                    "prediction": prediction,
                    "correct": is_correct,
                    **generation,
                }
            )
            if args.progress:
                print(
                    f"[{index + 1}/{source_count}] accuracy={correct / (local_index + 1):.3f} "
                    f"id={row['_id']} expected={expected}",
                    flush=True,
                )
    finally:
        if hook is not None and callable(getattr(hook, "remove", None)):
            hook.remove()
    return {
        "evaluator_schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_git_revision": _git_revision(),
        "evaluator_source_sha256": _source_sha256(),
        "benchmark": "LongBench-v2",
        "dataset_id": DATASET_ID,
        "dataset_split": DATASET_SPLIT,
        "dataset_split_role": "benchmark_test_only",
        "subset": args.subset,
        "difficulty": args.difficulty,
        "profile": f"longbench_v2_{args.subset}_{args.difficulty}",
        "instruction": "official_zero_shot_cot_with_explicit_choice_line" if args.thinking else "official_zero_shot_with_explicit_choice_line",
        "expected_rows": args.expected_rows,
        "source_count": source_count,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model": args.model,
        "model_path_real": os.path.realpath(args.model),
        "data": str(args.data.resolve()) if args.data is not None else f"hf://{DATASET_ID}/{DATASET_SPLIT}",
        "kv_bits": args.kv_bits,
        "quantizer": "official_kivi_packed_cache" if args.kivi else "qat_kv",
        "kivi_group_size": args.kivi_group_size if args.kivi else None,
        "kivi_residual_length": args.kivi_residual_length if args.kivi else None,
        "dtype": args.dtype,
        "dtype_resolved": dtype_resolved,
        "device": str(device),
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "preserve_projection_bias": args.preserve_projection_bias,
        "exact_bias_free_recompute": args.exact_bias_free_recompute,
        "seed": args.seed,
        "thinking_enabled": bool(args.thinking),
        "prompt_token_limit": None,
        "prompt_truncation": "none",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": 1,
            "do_sample": True,
        },
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "total_samples": len(rows),
        "truncated_samples": truncated,
        "truncated_fraction": truncated / len(rows) if rows else 0.0,
        "eos_samples": eos_samples,
        "eos_fraction": eos_samples / len(rows) if rows else 0.0,
        "extraction_failures": extraction_failures,
        "extraction_failure_fraction": extraction_failures / len(rows) if rows else 0.0,
        "mean_prompt_tokens": sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0.0,
        "max_prompt_tokens": max(prompt_counts, default=0),
        "mean_response_tokens": sum(response_counts) / len(response_counts) if response_counts else 0.0,
        "max_response_tokens": max(response_counts, default=0),
        "elapsed_seconds": round(time.time() - started, 3),
        "predictions": predictions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="local LongBench JSON/JSONL/Parquet; omit to use the pinned HF dataset",
    )
    parser.add_argument("--subset", choices=("short", "medium", "long", "all"), default=DEFAULT_SUBSET)
    parser.add_argument("--difficulty", choices=("easy", "hard", "all"), default=DEFAULT_DIFFICULTY)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--kv-bits", type=int, choices=(2, 4, 8, 16, 32), default=4)
    parser.add_argument("--preserve-projection-bias", action="store_true", default=True)
    parser.add_argument("--no-preserve-projection-bias", dest="preserve_projection_bias", action="store_false")
    parser.add_argument("--exact-bias-free-recompute", action="store_true")
    parser.add_argument("--kivi", action="store_true", help="use KIVI asymmetric KV2 fake quantization")
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=32)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking", dest="thinking", action="store_true")
    thinking_group.add_argument("--no-thinking", dest="thinking", action="store_false")
    # LongBench v2 uses its own short-answer prompt; do not add a separate
    # Thinking instruction unless a caller explicitly requests it.
    parser.set_defaults(thinking=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None and args.seed < 0:
        args.seed = None
    if args.num_shards <= 0 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("shard-index must satisfy 0 <= shard-index < num-shards, with num-shards > 0")
    if args.kivi and args.kv_bits != 2:
        raise SystemExit("--kivi requires --kv-bits 2")
    if args.max_new_tokens <= 0 or args.max_new_tokens > DEFAULT_MAX_NEW_TOKENS:
        raise SystemExit("max-new-tokens must be in (0,4096] for the LongRLVR evaluation contract")
    if not (0.0 < args.temperature) or not (0.0 < args.top_p <= 1.0) or args.top_k < -1:
        raise SystemExit("temperature must be positive, 0 < top-p <= 1, top-k >= -1")
    report = evaluate(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
