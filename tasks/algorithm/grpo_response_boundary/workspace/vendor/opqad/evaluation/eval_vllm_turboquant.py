"""Evaluate the two requested benchmarks with vLLM's native TurboQuant KV cache.

This intentionally keeps benchmark loading/scoring in the existing evaluators;
only model execution is replaced by vLLM so the cache dtype is the one used by
the rollout engine.
"""

from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MATH_BENCHMARKS = {"aime_math"}


def _chat_prompt(tokenizer: Any, content: str, *, thinking: bool = True) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


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


def _oscar_budget_utilization(args: argparse.Namespace) -> float:
    """gpu_memory_utilization with room left for the BF16 staging arena."""
    if not str(args.kv_cache_dtype).startswith("oscar"):
        return args.gpu_memory_utilization
    try:
        from opqad.runtime.oscar_stage_budget import adjust_utilization

        return adjust_utilization(
            args.model,
            args.max_num_seqs,
            args.gpu_memory_utilization,
            sink_tokens=int(os.environ.get("VLLM_OSCAR_SINK_TOKENS", "64")),
            recent_tokens=int(os.environ.get("VLLM_OSCAR_RECENT_TOKENS", "256")),
            block_size=int(os.environ.get("VLLM_OSCAR_BLOCK_SIZE", "16")),
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[oscar-budget] skipped: {type(exc).__name__}: {exc}", flush=True)
        return args.gpu_memory_utilization


def _engine_config(llm: Any) -> Any:
    """The live ``VllmConfig``, or None if this vLLM exposes it elsewhere.

    Read from the engine rather than reconstructed from argv, because the two
    disagree: ``arg_utils.py:1714`` unions OSCAR's boundary-skip layers into
    ``kv_cache_dtype_skip_layers`` *after* the caller's arguments are parsed, so
    a config rebuilt from ``args`` would show an empty skip list for a run that
    actually kept four layers in BF16.  That disagreement is exactly the bug
    this record exists to make visible.
    """
    for path in (
        ("llm_engine", "vllm_config"),
        ("llm_engine", "engine_core", "vllm_config"),
        ("engine", "vllm_config"),
    ):
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _oscar_record(args: argparse.Namespace, llm: Any) -> dict[str, Any] | None:
    """OPQAD  oscar record. K2V3/OSCAR paper implementation."""
    if not str(args.kv_cache_dtype).startswith("oscar"):
        return None
    skip: list[str] | None = None
    num_layers: int | None = None
    cfg = _engine_config(llm)
    if cfg is not None:
        cache = getattr(cfg, "cache_config", None)
        raw = getattr(cache, "kv_cache_dtype_skip_layers", None)
        if raw is not None:
            skip = [str(x) for x in raw]
        model = getattr(cfg, "model_config", None)
        text = getattr(model, "hf_text_config", None)
        num_layers = getattr(text, "num_hidden_layers", None)

    def _env(name: str, default: str = "") -> str:
        return os.environ.get(name, default).strip()

    sink = int(_env("VLLM_OSCAR_SINK_TOKENS", "64") or 64)
    recent = int(_env("VLLM_OSCAR_RECENT_TOKENS", "256") or 256)
    rec: dict[str, Any] = {
        "skip_layers": skip,
        "skip_layer_count": len(skip) if skip is not None else None,
        "num_hidden_layers": num_layers,
        "quant_all_layers_requested": True,
        "extra_skip_layers": _env("VLLM_OSCAR_EXTRA_SKIP_LAYERS"),
        "sink_tokens": sink,
        "recent_tokens": recent,
        "k_clip_ratio": float(_env("VLLM_OSCAR_K_CLIP_RATIO", "0") or 0),
        "v_clip_ratio": float(_env("VLLM_OSCAR_V_CLIP_RATIO", "0") or 0),
        "k_rotation_path": _env("VLLM_OSCAR_K_ROTATION_PATH"),
        "v_rotation_path": _env("VLLM_OSCAR_V_ROTATION_PATH"),
        "rotation_tag": os.path.basename(os.path.dirname(_env("VLLM_OSCAR_K_ROTATION_PATH"))) or None,
        "block_size": int(_env("VLLM_OSCAR_BLOCK_SIZE", "16") or 16),
        "per_request_staging": _env("VLLM_OSCAR_PER_REQUEST_STAGING") == "1",
        "stage_max_seqs": _env("VLLM_OSCAR_STAGE_MAX_SEQS") or None,
        "gqa_decode": _env("VLLM_OSCAR_GQA_DECODE", "1") == "1",
        "bf16_tail_mode": _env("VLLM_OSCAR_BF16_TAIL_MODE") or None,
        "enforce_eager": bool(getattr(args, "enforce_eager", False)),
        "storage": "packed INT2 + fp16 (scale, zero) per 128-element vector",
    }
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kv_bits import OSCAR_INT2

        scheme = replace(OSCAR_INT2, sink_tokens=sink, recent_tokens=recent)
        if num_layers and skip is not None:
            rec["effective_bits"] = round(scheme.effective_bits(args.max_model_len, num_layers, len(skip)), 4)
            rec["effective_bits_context"] = args.max_model_len
        else:
            rec["effective_bits"] = None
            rec["effective_bits_note"] = (
                "engine config did not expose kv_cache_dtype_skip_layers or num_hidden_layers; rate not computable from this run"
            )
    except Exception as exc:
        rec["effective_bits"] = None
        rec["effective_bits_note"] = f"{type(exc).__name__}: {exc}"
    return rec


def _make_llm(args: argparse.Namespace):
    from vllm import LLM
    # These patches target the optional OSCAR backend. Importing them for BF16
    # or native NVFP4 fails in environments where the OSCAR backend is absent.
    if str(args.kv_cache_dtype).startswith("oscar"):
        from opqad.runtime.oscar_cudagraph_patch import apply as _apply_oscar_cg

        _apply_oscar_cg()
        from opqad.runtime.oscar_v_stride_patch import apply as _apply_oscar_vstride

        _apply_oscar_vstride()
        from opqad.runtime.oscar_staging_patch import maybe_apply as _apply_oscar_pr_stage

        _apply_oscar_pr_stage()
        from opqad.runtime.oscar_gqa_patch import maybe_apply as _apply_oscar_gqa

        _apply_oscar_gqa()
        from opqad.runtime.oscar_quantile_patch import maybe_apply as _apply_oscar_quantile

        _apply_oscar_quantile()
        from opqad.runtime.oscar_layer_skip_patch import maybe_apply as _apply_oscar_all_layers

        _apply_oscar_all_layers()
    try:
        import torch

        if torch.cuda.is_initialized():
            print(
                "[eval] CUDA is already initialized in this process, so vLLM will spawn (not fork) the EngineCore.  The patches above are parent-only; the qat_oscar plugin re-applies them in the worker -- look for a [oscar-plugin] line with a different pid.",
                flush=True,
            )
    except Exception:
        pass
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": _oscar_budget_utilization(args),
        "enforce_eager": os.environ.get("EVAL_ENFORCE_EAGER", "1") != "0",
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if getattr(args, "compilation_config", None) is not None:
        kwargs["compilation_config"] = args.compilation_config
    if getattr(args, "kv_cache_memory_bytes", None) is not None:
        # Native vLLM cache-size control for measured physical-memory budgets.
        kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_memory_bytes)
    if getattr(args, "enable_metrics", False):
        kwargs["disable_log_stats"] = False
    if getattr(args, "language_model_only", False):
        kwargs["language_model_only"] = True
    extra_skip = os.environ.get("VLLM_OSCAR_EXTRA_SKIP_LAYERS", "").strip()
    if extra_skip:
        layers = sorted({s.strip() for s in extra_skip.split(",") if s.strip()})
        for s in layers:
            if not s.isdigit():
                raise SystemExit(
                    f"VLLM_OSCAR_EXTRA_SKIP_LAYERS: {s!r} is not a layer index; expected a comma-separated list like '4,5,8,10,11'"
                )
        kwargs["kv_cache_dtype_skip_layers"] = layers
        print(
            f"[eval] keeping layers {','.join(layers)} in native dtype (in addition to OSCAR's boundary layers)",
            flush=True,
        )
    _block_size = int(os.environ.get("VLLM_OSCAR_BLOCK_SIZE", "0") or 0)
    if _block_size:
        kwargs["block_size"] = _block_size
        print(f"[eval] block_size={_block_size}", flush=True)
    if args.calculate_kv_scales:
        kwargs["calculate_kv_scales"] = True
    if args.disable_q_quantization:
        kwargs["attention_config"] = {"disable_flashinfer_q_quantization": True}
    _backend = os.environ.get("QAT_FORCE_ATTENTION_BACKEND", "").strip()
    if _backend:
        cfg = dict(kwargs.get("attention_config") or {})
        cfg["backend"] = _backend
        kwargs["attention_config"] = cfg
        print(f"[eval] attention backend pinned to {_backend}", flush=True)
    return LLM(**kwargs)


def _sampling(args: argparse.Namespace):
    from vllm import SamplingParams

    if args.benchmark in MATH_BENCHMARKS:
        return SamplingParams(
            n=args.n, temperature=0.7, top_p=0.95, top_k=0, max_tokens=32768, stop_token_ids=getattr(args, "stop_token_ids", [151645, 151643]), seed=args.seed
        )
    return SamplingParams(
        n=1, temperature=0.1, top_p=1.0, max_tokens=4096, stop_token_ids=getattr(args, "stop_token_ids", [151645, 151643]), seed=args.seed
    )


def _load_prompts(args: argparse.Namespace, tokenizer: Any):
    if args.benchmark == "aime_math":
        from opqad.evaluation.eval_aime import DEFAULT_INSTRUCTION, answers_equal, extract_answer, load_aime_records

        records = load_aime_records(Path(args.data), args.limit)
        _no_instr = os.environ.get("EVAL_NO_INSTRUCTION", "0") == "1"
        prompts = [
            _chat_prompt(
                tokenizer, str(row["problem"]) if _no_instr else f"{DEFAULT_INSTRUCTION}\n\n{str(row['problem'])}", thinking=not getattr(args, "disable_thinking", False)
            )
            for row in records
        ]
        if _no_instr:
            print("[eval] EVAL_NO_INSTRUCTION=1: bare problems, no instruction prefix", flush=True)
        return (records, prompts, extract_answer, answers_equal)
    raise ValueError("Only AIME is supported by this evaluator")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import vllm

    if args.kv_cache_dtype in ("nvfp4", "fp8_e4m3") or str(args.kv_cache_dtype).startswith("turboquant_"):
        # The local packed-cache audit sends our trusted inspection callable to
        # EngineCore. vLLM requires this opt-in for callable worker RPCs.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = _make_llm(args)
    native_fp8_kv_audit = None
    if args.kv_cache_dtype == "fp8_e4m3":
        from opqad.runtime.native_fp8_verify import verify_native_fp8_rollout
        fp8_num_layers = int(json.loads((Path(args.model) / "config.json").read_text())["num_hidden_layers"])
        native_fp8_kv_audit = {"before_generation": verify_native_fp8_rollout(llm, fp8_num_layers)}
    native_nvfp4_kv_audit = None
    native_turboquant_kv_audit = None
    if args.kv_cache_dtype == "nvfp4":
        from opqad.runtime.native_tq_verify import verify_native_nvfp4_rollout

        # Fail before generation unless the live engine proves packed NVFP4 KV.
        native_nvfp4_kv_audit = verify_native_nvfp4_rollout(llm)
    if str(args.kv_cache_dtype).startswith("turboquant_"):
        from opqad.runtime.native_tq_verify import verify_native_turboquant_rollout
        model_config = json.loads((Path(args.model) / "config.json").read_text())
        native_turboquant_kv_audit = verify_native_turboquant_rollout(
            llm, args.kv_cache_dtype,
            num_hidden_layers=int(model_config["num_hidden_layers"]),
        )
    tokenizer = llm.get_tokenizer()
    records, prompts, extract_answer, aime_equal = _load_prompts(args, tokenizer)
    global_index = list(range(len(records)))
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")
        keep = list(range(args.shard, len(records), args.num_shards))
        global_index = keep
        records = [records[i] for i in keep]
        prompts = [prompts[i] for i in keep]
        print(
            f"[eval] shard {args.shard}/{args.num_shards}: {len(records)} problems (indices {keep[:4]}{('...' if len(keep) > 4 else '')}). Merge with opqad/evaluation/merge_eval_shards.py before reading accuracy.",
            flush=True,
        )
    if getattr(args, "tokenized_chat_prompts", False):
        # Templates already contain BOS/turn markers; preserve their token IDs.
        prompts = [{"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)} for p in prompts]
    sampling = _sampling(args)
    started = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=args.progress)
    if native_fp8_kv_audit is not None:
        native_fp8_kv_audit["after_generation"] = verify_native_fp8_rollout(llm, fp8_num_layers)
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
                str(row["answer"]).strip() if args.benchmark in MATH_BENCHMARKS else str(row["answer"]).strip().upper()
            )
            is_correct = bool(aime_equal(prediction, expected)) if aime_equal is not None else prediction == expected
            stats = _request_stats(request, output)
            prompt_counts.append(stats["prompt_tokens"])
            response_counts.append(stats["response_tokens"])
            truncated += int(stats["truncated"])
            eos += int(str(stats["finish_reason"]).lower() == "stop")
            extraction_failures += int(prediction is None)
            correct += int(is_correct)
            total_samples += 1
            row_predictions.append(
                {"sample": sample_index, "prediction": prediction, "correct": is_correct, "response": text, **stats}
            )
        predictions.append(
            {
                "index": global_index[index],
                "id": row.get("id", row.get("_id")),
                "answer": str(row["answer"]).strip(),
                "samples": row_predictions,
                "correct_samples": sum((int(x["correct"]) for x in row_predictions)),
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
        "data": str(Path(args.data).resolve()),
        "thinking_enabled": args.benchmark in MATH_BENCHMARKS and not getattr(args, "disable_thinking", False),
        "language_model_only": bool(getattr(args, "language_model_only", False)),
        "tokenized_chat_prompts": bool(getattr(args, "tokenized_chat_prompts", False)),
        "prompt_token_limit": None,
        "prompt_truncation": "none",
        "kv_cache_dtype_requested": args.kv_cache_dtype,
        "calculate_kv_scales": bool(args.calculate_kv_scales),
        "disable_q_quantization": args.disable_q_quantization,
        "kv_cache_quantization": "none" if args.kv_cache_dtype == "auto" else "vllm_native_" + args.kv_cache_dtype,
        "oscar": _oscar_record(args, llm),
        "native_fp8_kv_audit": native_fp8_kv_audit,
        "native_nvfp4_kv_audit": native_nvfp4_kv_audit,
        "native_turboquant_kv_audit": native_turboquant_kv_audit,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "sampling": {
            "temperature": 0.7 if args.benchmark in MATH_BENCHMARKS else 0.1,
            "top_p": 0.95 if args.benchmark in MATH_BENCHMARKS else 1.0,
            "n": args.n if args.benchmark in MATH_BENCHMARKS else 1,
            "top_k": 0 if args.benchmark in MATH_BENCHMARKS else None,
            "max_tokens": 32768 if args.benchmark in MATH_BENCHMARKS else 4096,
            "stop_token_ids": getattr(args, "stop_token_ids", [151645, 151643]),
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
        "shard": args.shard,
        "num_shards": args.num_shards,
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
    parser.add_argument("--benchmark", choices=("aime_math",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "nvfp4", "fp8_e4m3", "turboquant_k2v3_nc", "turboquant_k3v2_nc", "turboquant_3bit_nc", "oscar_int2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--disable-q-quantization",
        action="store_true",
        help="keep queries in the model dtype when the KV cache is quantized",
    )
    parser.add_argument("--stop-token-ids", type=int, nargs="+", default=[151645, 151643], help="Model-specific end-of-turn/end-of-text token IDs.")
    parser.add_argument("--disable-thinking", action="store_true", help="Use the ordinary model chat template; do not request a thinking mode.")
    parser.add_argument("--tokenized-chat-prompts", action="store_true", help="Pass rendered-template token IDs directly to vLLM without adding special tokens.")
    parser.add_argument("--language-model-only", action="store_true", help="Load only the language model of a multimodal checkpoint for text evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calculate-kv-scales",
        action="store_true",
        help="compute KV scales at runtime instead of defaulting to 1.0; required for a fair reading of calibration-dependent dtypes",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(args)
        if report.get("total_samples", 0) > 0 and report.get("extraction_failures", 0) == report.get("total_samples"):
            report["status"] = "invalid"
            report["error"] = "all generated samples failed answer extraction"
            rc = 2
        else:
            report["status"] = "passed"
            rc = 0
    except Exception as exc:
        report = {"evaluator_schema_version": 2, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        audit = getattr(exc, "record", None)
        if audit is not None:
            report["native_nvfp4_kv_audit"] = audit
        rc = 1
    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
