#!/usr/bin/env python3
"""Evaluate a local causal LM on the 30 AIME 2024 problems.

This is deliberately a small, dependency-light evaluator for the QAT/OPD
experiments.  It uses the model's normal HF generation path (therefore the
same sampling policy as rollout), and can install the repository's KV
fake-quantization hooks on the student.  ``top_k`` here is a *generation*
filter; it is independent from the top-k token set used by MOPD's reward.

Example::

  python scripts/eval_aime.py \
    --model /historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b \
    --kv-bits 2 --profile aime_math --num-samples 4 \
    --output /historical/shared/qat/runs/aime/kv4.json

The evaluator does not require a critic or a reference model.  For a short
smoke run use ``--limit 1 --max-new-tokens 32``.  A modern Transformers
version (4.37+; 4.45+ recommended for Qwen2) is required to load the model
configs; this requirement is checked with a clear error instead of silently
falling back to the old LLaMA implementation in this repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_AIME = Path("/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet")
DEFAULT_INSTRUCTION = (
    "Solve the following AIME problem. Think step by step, explain your reasoning, "
    "and finish with the integer answer in \\boxed{}."
)


def load_aime_records(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read AIME parquet rows without requiring the datasets library."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("AIME evaluation requires pyarrow") from exc
    rows = pq.read_table(path).to_pylist()
    result: List[Dict[str, Any]] = []
    for row in rows[: limit if limit is not None else None]:
        if not isinstance(row, dict) or "problem" not in row or "answer" not in row:
            raise ValueError(f"invalid AIME row in {path}: {row!r}")
        result.append(row)
    return result


def extract_answer(text: str) -> Optional[str]:
    """Extract an AIME integer, preferring boxed/final-answer conventions."""

    if not text:
        return None
    # Models use both LaTex \u005cboxed{123} and ``\\boxed{\\text{123}}``.
    boxed = re.findall(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?(-?\d{1,3})", text)
    if boxed:
        return boxed[-1].lstrip("+")
    marked = re.findall(
        r"(?:final answer|answer|答案)\s*(?:is|=|：|:)\s*[^0-9-]*(-?\d{1,3})",
        text,
        flags=re.IGNORECASE,
    )
    if marked:
        return marked[-1].lstrip("+")
    # Last standalone 0--999 integer is a conservative fallback.  Restricting
    # to three digits avoids selecting years or intermediate large constants.
    numbers = re.findall(r"(?<![\w])-?\d{1,3}(?![\w])", text)
    return numbers[-1].lstrip("+") if numbers else None


def answers_equal(prediction: Optional[str], expected: str) -> bool:
    """Compare AIME answers numerically (``025`` and ``25`` are equivalent)."""

    if prediction is None:
        return False
    try:
        return 0 <= int(prediction) <= 999 and int(prediction) == int(str(expected).strip())
    except (TypeError, ValueError):
        return prediction.strip() == str(expected).strip()


def _messages(problem: str, instruction: str = DEFAULT_INSTRUCTION) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{instruction}\n\n{problem}"}]


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


def _git_revision() -> Optional[str]:
    """Return the evaluator checkout revision when it is available.

    Evaluation reports are often copied away from the source checkout.  A
    best-effort revision makes those reports auditable without making a
    benchmark depend on git being installed (or on the workspace being a
    repository).  No error from this diagnostic should affect evaluation.
    """

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
    """Hash this evaluator source for provenance of uncommitted changes."""

    try:
        digest = hashlib.sha256()
        with Path(__file__).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _resolved_dtype_name(dtype: torch.dtype) -> str:
    """Map the actual torch dtype to a stable report spelling."""

    names = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return names.get(dtype, str(dtype).replace("torch.", ""))


def _load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("AIME evaluation requires transformers") from exc
    version = tuple(int(part) for part in re.match(r"^(\d+)\.(\d+)", transformers.__version__).groups())
    if version < (4, 37):
        raise RuntimeError(
            f"Transformers {transformers.__version__} is too old for Qwen2; install >=4.37"
        )
    # ``auto`` traditionally follows the CUDA default and therefore chooses
    # BF16.  That is appropriate for KV2/KV4 diagnostics, but it would make
    # the supposedly full-precision (KV32) reference silently BF16.  Keep
    # the baseline unambiguous: callers can still explicitly request BF16 for
    # a memory diagnostic with ``--dtype bf16``.
    requested_dtype_name = args.dtype
    if args.kv_bits >= 32 and args.dtype == "auto":
        requested_dtype_name = "fp32"
    dtype = _dtype(requested_dtype_name, device)
    # ``auto`` is only a request; report the dtype actually selected (BF16 on
    # CUDA in the normal gate environment) so precision provenance is explicit.
    resolved_dtype_name = _resolved_dtype_name(dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("AIME evaluation requires a tokenizer with pad_token_id")
    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    # ``attn_implementation`` was added after the earliest supported
    # Transformers releases; omitting it when unset keeps the script portable.
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device).eval()
    hook = None
    if getattr(args, "naive_kv", False):
        from qat_kv.naive_kv_fake_quant import install_naive_kv_fake_quant

        hook = install_naive_kv_fake_quant(
            bits=args.kv_bits,
            group_size=args.naive_kv_group_size,
            symmetric=args.naive_kv_symmetric,
            value_residual_length=args.naive_kv_value_residual,
        )
    elif args.kivi and args.kivi_fake:
        # Functional quantise/dequantise instead of the packed cache: the
        # tensors stay bf16 so attention keeps its normal fast path, which is
        # what makes this variant usable for accuracy sweeps.  It saves no
        # memory -- that is the packed implementation's job -- so never quote
        # a footprint from a --kivi-fake run.
        from qat_kv.kivi_fake_quant import install_kivi_fake_quant

        hook = install_kivi_fake_quant(
            bits=2,
            group_size=args.kivi_group_size,
            residual_length=args.kivi_residual_length,
        )
    elif args.kivi:
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
        )
    return model, tokenizer, hook, resolved_dtype_name


def _encode_prompt(tokenizer, messages: Sequence[Dict[str, str]], device: torch.device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return ids.to(device)


def _sampling(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve paper-evaluation versus training-rollout defaults.

    The OPD papers use different decoding policies for training and
    benchmarking.  Keeping these presets explicit prevents accidentally
    reporting a training sampler as a benchmark number.
    """

    defaults = {
        # Math test evaluation: no prompt cap is applied by this evaluator and
        # the response budget is the requested 32K tokens.  The chat template
        # for DeepSeek-R1-Distill-Qwen adds the thinking marker; the instruction
        # also makes the contract explicit in the generated prompt.
        "aime_math": {"temperature": 0.7, "top_k": -1, "top_p": 0.95,
                      "max_new_tokens": 32768, "num_samples": 4},
        "paper": {"temperature": 0.7, "top_k": -1, "top_p": 0.95,
                  "max_new_tokens": 32768, "num_samples": 4},
        # LongRLVR's RL/long-context recipe keeps unrestricted nucleus
        # sampling (top-p=1, top-k disabled), temperature 1 and 4096-token
        # responses.  This is useful for the corresponding ablation.
        "longrlvr": {"temperature": 1.0, "top_k": -1, "top_p": 1.0,
                     "max_new_tokens": 4096, "num_samples": 1},
        # Match the student rollout policy used by the MOPD launcher.  The
        # reward-support top-k (64) is a separate training signal and must not
        # be reused as a generation filter here.
        "training": {"temperature": 1.0, "top_k": -1, "top_p": 0.99,
                      "max_new_tokens": 4096, "num_samples": 1},
    }[args.profile]
    return {
        key: getattr(args, key) if getattr(args, key) is not None else value
        for key, value in defaults.items()
    }


@torch.inference_mode()
def _generate_samples(
    model,
    tokenizer,
    problem: str,
    args: argparse.Namespace,
    device: torch.device,
    sampling: Dict[str, Any],
    num_return_sequences: int = 1,
) -> List[Dict[str, Any]]:
    """Generate ``num_return_sequences`` completions for one problem at once.

    avg@N draws N independent samples from the same prompt, so they can share a
    single generate() call.  Batching only within a problem keeps the batch
    free of padding -- every row is the same prompt, hence the same length --
    which matters for KV quantisation, whose group statistics would otherwise
    be computed partly over pad positions.

    Decode on this evaluator is memory-bound at batch 1, so the extra rows are
    close to free; a 32K-token budget over 30 problems x N samples is otherwise
    sequential enough to outrun an 8-hour wall limit.
    """

    input_ids = _encode_prompt(tokenizer, _messages(problem, args.instruction), device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=sampling["temperature"],
        # vLLM/Open-MOPD uses -1 to mean disabled, while HF's
        # TopKLogitsWarper uses 0 for the same setting.
        top_k=max(0, sampling["top_k"]),
        top_p=sampling["top_p"],
        max_new_tokens=sampling["max_new_tokens"],
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **({"cache_implementation": "dynamic"} if getattr(args, "kivi", False) else {}),
    )
    prompt_tokens = int(input_ids.shape[-1])
    eos_token_id = tokenizer.eos_token_id
    results: List[Dict[str, Any]] = []
    for row in range(int(output.shape[0])):
        completion = output[row, prompt_tokens:]
        # generate() runs until every row in the batch has stopped and pads the
        # rows that finished earlier.  Cutting at the first EOS reproduces what
        # this row would have produced on its own, so response_tokens and the
        # finish reason do not depend on how the batch was packed.
        eos_seen = False
        if eos_token_id is not None and completion.numel():
            hit = torch.eq(completion, int(eos_token_id)).nonzero()
            if hit.numel():
                completion = completion[: int(hit[0, 0]) + 1]
                eos_seen = True
        generated_tokens = int(completion.numel())
        hit_max_tokens = generated_tokens >= int(sampling["max_new_tokens"])
        if eos_seen:
            finish_reason = "eos"
        elif hit_max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stopped"
        results.append({
            "response": tokenizer.decode(completion, skip_special_tokens=True),
            "prompt_tokens": prompt_tokens,
            "prompt_truncated": False,
            "response_tokens": generated_tokens,
            "eos_seen": eos_seen,
            "hit_max_tokens": hit_max_tokens,
            # A response that reaches the configured budget without EOS is the
            # operational definition used by this evaluator for truncation.
            "truncated": bool(hit_max_tokens and not eos_seen),
            "finish_reason": finish_reason,
        })
    return results


@torch.inference_mode()
def _generate_with_metadata(
    model,
    tokenizer,
    problem: str,
    args: argparse.Namespace,
    device: torch.device,
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate one completion and retain termination metadata.

    Benchmark accuracy alone cannot distinguish a genuinely wrong answer from
    a response that simply exhausted the generation budget.  Keep the raw
    completion length and EOS status next to the decoded text so intermediate
    AIME checks remain interpretable.
    """
    return _generate_samples(model, tokenizer, problem, args, device, sampling, 1)[0]


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    problem: str,
    args: argparse.Namespace,
    device: torch.device,
    sampling: Dict[str, Any],
) -> str:
    """Backward-compatible text-only generation helper."""

    return _generate_with_metadata(model, tokenizer, problem, args, device, sampling)["response"]


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    # ``torch.device('auto')`` is not a valid device.  Resolve the sentinel
    # explicitly so CPU-only smoke/evaluator runs remain useful when CUDA is
    # unavailable (and so a requested CUDA device still fails with a clear
    # error below).
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
    records = load_aime_records(Path(args.data), args.limit)
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    source_count = len(records)
    indexed_records = list(enumerate(records))
    if args.num_shards > 1:
        indexed_records = indexed_records[args.shard_index :: args.num_shards]
    records = [record for _, record in indexed_records]
    sampling = _sampling(args)
    model, tokenizer, hook, resolved_dtype_name = _load_model_and_tokenizer(args, device)
    predictions = []
    correct = 0
    generated_token_counts: List[int] = []
    prompt_token_counts: List[int] = []
    truncated_samples = 0
    extraction_failures = 0
    eos_samples = 0
    finish_reason_counts: Dict[str, int] = {}
    started = time.time()
    try:
        for local_index, (index, row) in enumerate(indexed_records):
            sample_started = time.time()
            expected = str(row["answer"]).strip()
            samples = []
            # Decode is memory-bound here (measured: 44.9 ms/step at batch 1
            # vs 42.6 at batch 8), so drawing the avg@N samples together costs
            # almost nothing per step and cuts wall-clock by close to the batch
            # width.  Defaults to 1, leaving every other caller unchanged.
            per_batch = max(1, int(getattr(args, "batch_samples", 1) or 1))
            sample_index = 0
            while sample_index < sampling["num_samples"]:
                take = min(per_batch, sampling["num_samples"] - sample_index)
                for generation in _generate_samples(
                    model, tokenizer, str(row["problem"]), args, device, sampling, take
                ):
                    response = generation["response"]
                    prediction = extract_answer(response)
                    is_correct = answers_equal(prediction, expected)
                    generated_token_counts.append(int(generation["response_tokens"]))
                    prompt_token_counts.append(int(generation["prompt_tokens"]))
                    truncated_samples += int(generation["truncated"])
                    extraction_failures += int(prediction is None)
                    eos_samples += int(generation["eos_seen"])
                    finish_reason = str(generation["finish_reason"])
                    finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
                    samples.append(
                        {
                            "sample": sample_index,
                            "prediction": prediction,
                            "correct": is_correct,
                            **generation,
                        }
                    )
                    sample_index += 1
            sample_correct = sum(int(sample["correct"]) for sample in samples)
            # ``avg@N`` is the mean correctness over independent samples, as in
            # the OPD paper; it is not pass@N / majority voting.
            correct += sample_correct / sampling["num_samples"]
            predictions.append(
                {
                    "index": index,
                    "id": row.get("id"),
                    "year": row.get("year"),
                    "answer": expected,
                    "prediction": samples[0]["prediction"],
                    "correct": bool(samples[0]["correct"]),
                    "sample_correct": sample_correct,
                    "sample_accuracy": sample_correct / sampling["num_samples"],
                    "samples": samples,
                    "seconds": round(time.time() - sample_started, 3),
                }
            )
            if args.progress:
                print(f"[{index + 1}/{source_count}] avg@{sampling['num_samples']}="
                      f"{sample_correct / sampling['num_samples']:.3f} expected={expected}", flush=True)
    finally:
        if hook is not None and callable(getattr(hook, "remove", None)):
            hook.remove()
    report = {
        "evaluator_schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_git_revision": _git_revision(),
        "evaluator_source_sha256": _source_sha256(),
        "model": args.model,
        "model_path_real": os.path.realpath(args.model),
        "data": str(Path(args.data).resolve()),
        "kv_bits": args.kv_bits,
        "naive_kv": bool(getattr(args, "naive_kv", False)),
        "naive_kv_group_size": args.naive_kv_group_size if getattr(args, "naive_kv", False) else None,
        "naive_kv_value_residual": args.naive_kv_value_residual if getattr(args, "naive_kv", False) else None,
        "quantizer": ("naive_per_token" if getattr(args, "naive_kv", False)
                      else "kivi_fake_quant" if args.kivi and args.kivi_fake
                      else "official_kivi_packed_cache" if args.kivi else "qat_kv"),
        "kivi_group_size": args.kivi_group_size if args.kivi else None,
        "kivi_residual_length": args.kivi_residual_length if args.kivi else None,
        # ``dtype`` is the user request; ``dtype_resolved`` records the actual
        # torch dtype policy.  In particular, KV32 + dtype=auto is FP32.
        "dtype": args.dtype,
        "dtype_resolved": resolved_dtype_name,
        "device": str(device),
        "attn_implementation": args.attn_implementation,
        "preserve_projection_bias": args.preserve_projection_bias,
        "seed": args.seed,
        "instruction": args.instruction,
        "thinking_enabled": bool(args.thinking),
        "prompt_token_limit": None,
        "prompt_truncation": "none",
        "prompt_tokens_min": min(prompt_token_counts, default=0),
        "prompt_tokens_max": max(prompt_token_counts, default=0),
        "profile": args.profile,
        "source_count": source_count,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sampling": {**sampling, "hf_top_k": max(0, sampling["top_k"]), "do_sample": True},
        "count": len(records),
        # For num_samples=1 this is the ordinary number correct; for avg@N it
        # is the sum of per-problem sample accuracies (hence may be fractional).
        "correct": correct,
        "correct_samples_normalized": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "total_samples": len(generated_token_counts),
        "truncated_samples": truncated_samples,
        "truncated_fraction": (
            truncated_samples / len(generated_token_counts) if generated_token_counts else 0.0
        ),
        "mean_response_tokens": (
            sum(generated_token_counts) / len(generated_token_counts) if generated_token_counts else 0.0
        ),
        "max_response_tokens": max(generated_token_counts, default=0),
        "extraction_failures": extraction_failures,
        "extraction_failure_fraction": (
            extraction_failures / len(generated_token_counts) if generated_token_counts else 0.0
        ),
        "eos_samples": eos_samples,
        "eos_fraction": eos_samples / len(generated_token_counts) if generated_token_counts else 0.0,
        "finish_reason_counts": finish_reason_counts,
        "elapsed_seconds": round(time.time() - started, 3),
        "predictions": predictions,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_AIME)
    parser.add_argument("--kv-bits", type=int, choices=(2, 4, 8, 16, 32), default=2)
    bias_group = parser.add_mutually_exclusive_group()
    bias_group.add_argument(
        "--preserve-projection-bias",
        dest="preserve_projection_bias",
        action="store_true",
        default=True,
        help="keep learned Qwen2 k/v projection bias in FP while quantizing its residual (default)",
    )
    bias_group.add_argument(
        "--no-preserve-projection-bias",
        dest="preserve_projection_bias",
        action="store_false",
        help="diagnostic: quantize the complete projection output",
    )
    parser.add_argument("--profile", choices=("aime_math", "paper", "longrlvr", "training"), default="aime_math",
                        help="aime_math: thinking prompt, no prompt cap, 32768-token response, avg@4; "
                             "paper: 32768-token benchmark preset; "
                             "longrlvr: temp 1/top-p 1/generation top-k disabled/4096; "
                             "training: temp 1/top-p .99/generation top-k disabled/4096 "
                             "(MOPD reward support top-k=64 is a separate training setting)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="independent samples per problem (math default 4)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--naive-kv", action="store_true",
                        help="LLM-QAT-style per-token min/max KV fake quantization "
                             "(the baseline KIVI and TurboQuant are measured against)")
    parser.add_argument("--naive-kv-group-size", type=int, default=None,
                        help="subdivide head_dim; default is one scale per token per head")
    parser.add_argument("--naive-kv-symmetric", action="store_true")
    parser.add_argument("--naive-kv-value-residual", type=int, default=0,
                        help="keep the most recent N values in full precision; on KIVI this\n                             tail alone accounts for the entire 41.67 vs 28.33 AIME gap")
    parser.add_argument("--kivi", action="store_true", help="use KIVI asymmetric KV2 fake quantization")
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=32,
                        help="0 removes KIVI's high-precision window entirely "
                             "(--kivi-fake only); the window is position-dependent "
                             "and therefore hostile to QAT")
    parser.add_argument("--kivi-fake", action="store_true",
                        help="use the functional fake quantiser instead of the packed cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed (set -1 to leave RNG unchanged)")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking", dest="thinking", action="store_true",
        help="record and request reasoning-mode evaluation (default for math profile)",
    )
    thinking_group.add_argument(
        "--no-thinking", dest="thinking", action="store_false",
        help="disable the explicit reasoning contract (diagnostic only)",
    )
    parser.set_defaults(thinking=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--batch-samples", type=int, default=1,
                        help="draw this many avg@N samples per generate() call "
                             "(HF path only); 1 reproduces one-at-a-time decoding")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_shards <= 0 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("shard-index must satisfy 0 <= shard-index < num-shards, with num-shards > 0")
    if getattr(args, "naive_kv", False) and args.kivi:
        raise SystemExit("--naive-kv and --kivi are mutually exclusive")
    if args.kivi and args.kv_bits != 2:
        raise SystemExit("--kivi requires --kv-bits 2")
    if args.seed is not None and args.seed < 0:
        args.seed = None
    sampling = _sampling(args)
    if not (0 < sampling["top_p"] <= 1) or sampling["top_k"] < -1 or sampling["max_new_tokens"] <= 0:
        raise SystemExit("require 0 < top-p <= 1, top-k >= -1, max-new-tokens > 0")
    if sampling["num_samples"] <= 0:
        raise SystemExit("num-samples must be positive")
    report = evaluate(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
