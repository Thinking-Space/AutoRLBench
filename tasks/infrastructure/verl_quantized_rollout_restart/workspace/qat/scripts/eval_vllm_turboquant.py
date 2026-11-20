#!/usr/bin/env python3
"""Evaluate the two requested benchmarks with vLLM's native TurboQuant KV cache.

This intentionally keeps benchmark loading/scoring in the existing evaluators;
only model execution is replaced by vLLM so the cache dtype is the one used by
the rollout engine.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATH_BENCHMARKS = {"aime_math", "amc23", "minerva_math", "olympiad_bench"}


def _chat_prompt(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _request_stats(request: Any, output: Any) -> dict[str, Any]:
    prompt_ids = getattr(request, "prompt_token_ids", None) or []
    token_ids = getattr(output, "token_ids", None) or []
    finish = getattr(output, "finish_reason", None)
    return {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(token_ids),
        "finish_reason": str(finish) if finish is not None else None,
        "truncated": str(finish).lower() == "length",
    }


def _make_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # Was hardcoded True.  EVAL_ENFORCE_EAGER=0 opts into vLLM's compiled +
        # CUDA-graph decode path, which measured 4.4x faster decode on this GPU
        # (14.45 -> 3.26 ms/step at batch 8).  Default is unchanged, so every
        # existing eval keeps its exact configuration.
        "enforce_eager": os.environ.get("EVAL_ENFORCE_EAGER", "1") != "0",
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.calculate_kv_scales:
        # vLLM loads k_scale/v_scale from the checkpoint and otherwise falls
        # back to 1.0.  A plain BF16 HF checkpoint carries no scales, so every
        # calibration-dependent cache dtype (fp8_e4m3, the per-token-head
        # int8/fp8 paths, nvfp4) was being measured at scale=1.0 -- which is
        # what produced token-level repetition rather than a graceful accuracy
        # loss.  This computes the scales on the fly so the dtype is judged on
        # its quantiser, not on a missing calibration artefact.
        kwargs["calculate_kv_scales"] = True
    if args.disable_q_quantization:
        # vllm/v1/attention/backends/flashinfer.py:649 quantizes the *query* to
        # fp8 whenever the KV cache is nvfp4 -- a second quantization the
        # --kv-cache-dtype flag does not name.  This turns it off so the KV
        # tier can be measured on its own.
        kwargs["attention_config"] = {"disable_flashinfer_q_quantization": True}
    # Qwen2.5-1M needs this field for its long-context dual-chunk attention.
    # It is deliberately not overridden: the benchmark must exercise the
    # original model configuration.
    return LLM(**kwargs)


def _sampling(args: argparse.Namespace):
    from vllm import SamplingParams

    # Qwen2 chat turns terminate with <|im_end|> (151645), while the model
    # config also lists <|endoftext|> (151643).  vLLM 0.21 may expose only the
    # latter as eos_token_id for this checkpoint; pass both explicitly or the
    # model continues into unrelated training-text until max_tokens.
    if args.benchmark in MATH_BENCHMARKS:
        return SamplingParams(
            n=args.n,
            temperature=0.7,
            top_p=0.95,
            max_tokens=32768,
            stop_token_ids=[151643],
            seed=args.seed,
        )
    return SamplingParams(
        n=1,
        temperature=0.1,
        top_p=1.0,
        max_tokens=4096,
        stop_token_ids=[151645, 151643],
        seed=args.seed,
    )


def _load_prompts(args: argparse.Namespace, tokenizer: Any):
    sys_path = str(ROOT)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    if args.benchmark == "aime_math":
        from scripts.eval_aime import (
            DEFAULT_INSTRUCTION,
            answers_equal,
            extract_answer,
            load_aime_records,
        )

        records = load_aime_records(Path(args.data), args.limit)
        prompts = [
            _chat_prompt(
                tokenizer,
                f"{DEFAULT_INSTRUCTION}\n\n{str(row['problem'])}",
            )
            for row in records
        ]
        return records, prompts, extract_answer, answers_equal

    if args.benchmark in {"amc23", "minerva_math", "olympiad_bench"}:
        from scripts.eval_math_benchmarks import (
            DEFAULT_INSTRUCTION,
            extract_math_answer,
            load_math_records,
            math_answers_equal,
        )

        records = load_math_records(Path(args.data), args.benchmark, args.limit)
        prompts = [
            _chat_prompt(
                tokenizer,
                f"{DEFAULT_INSTRUCTION}\n\n{str(row['problem'])}",
            )
            for row in records
        ]
        return (
            records,
            prompts,
            lambda text: extract_math_answer(text, args.benchmark),
            math_answers_equal,
        )

    from scripts.eval_longbench_v2 import (
        build_prompt,
        extract_answer,
        load_longbench_records,
    )

    records = load_longbench_records(
        Path(args.data) if args.data else None,
        subset="medium",
        difficulty="all",
        limit=args.limit,
        expected_rows=215 if args.limit is None else None,
    )
    prompts = [_chat_prompt(tokenizer, build_prompt(row, thinking=False)) for row in records]
    return records, prompts, extract_answer, None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import vllm

    llm = _make_llm(args)
    tokenizer = llm.get_tokenizer()
    records, prompts, extract_answer, aime_equal = _load_prompts(args, tokenizer)
    sampling = _sampling(args)
    started = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=args.progress)
    predictions: list[dict[str, Any]] = []
    correct = 0
    total_samples = 0
    truncated = 0
    eos = 0
    extraction_failures = 0
    prompt_counts: list[int] = []
    response_counts: list[int] = []

    for index, (row, request) in enumerate(zip(records, outputs)):
        row_outputs = list(getattr(request, "outputs", []) or [])
        row_predictions = []
        for sample_index, output in enumerate(row_outputs):
            text = str(getattr(output, "text", ""))
            prediction = extract_answer(text)
            expected = (
                str(row["answer"]).strip()
                if args.benchmark in MATH_BENCHMARKS
                else str(row["answer"]).strip().upper()
            )
            is_correct = (
                bool(aime_equal(prediction, expected))
                if aime_equal is not None
                else prediction == expected
            )
            stats = _request_stats(request, output)
            prompt_counts.append(stats["prompt_tokens"])
            response_counts.append(stats["response_tokens"])
            truncated += int(stats["truncated"])
            eos += int(str(stats["finish_reason"]).lower() == "stop")
            extraction_failures += int(prediction is None)
            correct += int(is_correct)
            total_samples += 1
            row_predictions.append(
                {
                    "sample": sample_index,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": text,
                    **stats,
                }
            )
        predictions.append(
            {
                "index": index,
                "id": row.get("id", row.get("_id")),
                "answer": str(row["answer"]).strip(),
                "samples": row_predictions,
                "correct_samples": sum(int(x["correct"]) for x in row_predictions),
            }
        )
        if args.progress:
            print(f"[{index + 1}/{len(records)}]", flush=True)

    return {
        "evaluator_schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "vllm_native_turboquant",
        "vllm_version": vllm.__version__,
        "vllm_compatibility": {
            "flashinfer_accepts_qwen_layer_idx": True,
            "flashinfer_accepts_dual_chunk_attention_config": True,
            "note": "constructor-only compatibility patch; cache/math unchanged",
        },
        "model": args.model,
        "model_path_real": os.path.realpath(args.model),
        "benchmark": args.benchmark,
        "data": str(Path(args.data).resolve()) if args.data else "hf://THUDM/LongBench-v2",
        "subset": "medium" if args.benchmark == "longbench_v2_medium" else None,
        "difficulty": "all" if args.benchmark == "longbench_v2_medium" else None,
        "thinking_enabled": args.benchmark in MATH_BENCHMARKS,
        "prompt_token_limit": None,
        "prompt_truncation": "none",
        "kv_cache_dtype_requested": args.kv_cache_dtype,
        "calculate_kv_scales": bool(args.calculate_kv_scales),
        "disable_q_quantization": args.disable_q_quantization,
        "kv_cache_quantization": (
            "none"
            if args.kv_cache_dtype == "auto"
            else (
                "vllm_native_turboquant"
                if args.kv_cache_dtype.startswith("turboquant")
                else f"vllm_native_{args.kv_cache_dtype}"
            )
        ),
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "sampling": {
            "temperature": 0.7 if args.benchmark in MATH_BENCHMARKS else 0.1,
            "top_p": 0.95 if args.benchmark in MATH_BENCHMARKS else 1.0,
            "n": args.n if args.benchmark in MATH_BENCHMARKS else 1,
            "max_tokens": 32768 if args.benchmark in MATH_BENCHMARKS else 4096,
            "stop_token_ids": [151643] if args.benchmark in MATH_BENCHMARKS else [151645, 151643],
            "seed": args.seed,
        },
        "count": len(records),
        "correct": correct,
        "total_samples": total_samples,
        "accuracy": correct / total_samples if total_samples else 0.0,
        "truncated_samples": truncated,
        "truncated_fraction": truncated / total_samples if total_samples else 0.0,
        "eos_samples": eos,
        "eos_fraction": eos / total_samples if total_samples else 0.0,
        "extraction_failures": extraction_failures,
        "mean_prompt_tokens": sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0.0,
        "max_prompt_tokens": max(prompt_counts, default=0),
        "mean_response_tokens": sum(response_counts) / len(response_counts) if response_counts else 0.0,
        "max_response_tokens": max(response_counts, default=0),
        "elapsed_seconds": round(time.time() - started, 3),
        "predictions": predictions,
    }


def _write_report(path: Path, report: dict[str, Any], attempts: int = 6) -> None:
    """Write a report atomically, retrying transient shared-filesystem errors.

    Evaluation can run for hours and the final rename/write occasionally sees
    ``EREMOTEIO`` on the shared filesystem.  Keeping the temporary file next
    to the requested output makes a successful retry atomic and avoids ever
    exposing a partial JSON report to downstream scripts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    last_error: OSError | None = None
    for attempt in range(attempts):
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{random.randrange(1 << 30):08x}")
        try:
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt + 1 < attempts:
                time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=(
            "aime_math",
            "amc23",
            "minerva_math",
            "olympiad_bench",
            "longbench_v2_medium",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=None)
    # "auto" is the unquantized control arm: vLLM keeps the cache in the
    # model dtype.  Without it there is no way to ask whether QAT cost the
    # checkpoint anything at full precision.
    parser.add_argument(
        "--kv-cache-dtype",
        # Whatever vllm.config.cache.CacheDType accepts in this build.  Note
        # int4_per_token_head is NOT in it; only int8/fp8 per-token-head and
        # nvfp4.  nvfp4 goes through flashinfer and may need a newer compute
        # capability than the L20D's 8.9, so it has to be tried, not assumed.
        choices=(
            "auto",
            "turboquant_3bit_nc", "turboquant_k3v4_nc", "turboquant_4bit_nc",
            # Kept in sync by hand with vllm.config.cache.CacheDType -- this is
            # a copy, not a reference, so a tier registered in vLLM is still
            # rejected here until it is added.  Job 8961 died on exactly that.
            "turboquant_2bit_nc", "turboquant_k2v3_nc",
            "int8_per_token_head", "fp8_per_token_head", "nvfp4",
            "fp8", "fp8_e4m3", "fp8_e5m2",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    # Samples per problem on math benchmarks.  30 AIME problems x n=4 gives a standard
    # error of ~4.5pp, which cannot resolve a few-point QAT effect; n=16
    # brings it to ~2.3pp.  Ignored by longbench_v2_medium, which is n=1.
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--disable-q-quantization",
        action="store_true",
        help="keep queries in the model dtype when the KV cache is quantized",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calculate-kv-scales", action="store_true",
                        help="compute KV scales at runtime instead of defaulting to 1.0; "
                             "required for a fair reading of calibration-dependent dtypes")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    # Fill in default data path if not provided
    if args.data is None:
        if args.benchmark == "aime_math":
            from scripts.eval_aime import DEFAULT_AIME
            args.data = DEFAULT_AIME
        elif args.benchmark in {"amc23", "minerva_math", "olympiad_bench"}:
            # These benchmarks also need default paths
            raise ValueError(f"--data is required for benchmark {args.benchmark}")
        # longbench_v2_medium uses HuggingFace dataset, args.data can stay None

    try:
        report = evaluate(args)
        # A completed vLLM request is not automatically a valid benchmark
        # result.  If every sample fails answer extraction (as happens when
        # a long-context cache configuration emits corrupted text), mark the
        # report invalid and fail closed so downstream training is not started
        # from an unusable evaluation gate.
        if (
            report.get("total_samples", 0) > 0
            and report.get("extraction_failures", 0) == report.get("total_samples")
        ):
            report["status"] = "invalid"
            report["error"] = "all generated samples failed answer extraction"
            rc = 2
        else:
            report["status"] = "passed"
            rc = 0
    except Exception as exc:
        report = {
            "evaluator_schema_version": 2,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        rc = 1
    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
