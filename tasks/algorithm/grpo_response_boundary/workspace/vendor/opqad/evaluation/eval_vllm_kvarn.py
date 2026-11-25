#!/usr/bin/env python3
"""KVarN packed K4V2: matched throughput probe and QAT AIME24 evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from opqad.integrations.kvarn_vllm import KVARN_REVISION, make_engine
from opqad.evaluation.eval_aime import DEFAULT_INSTRUCTION, answers_equal, extract_answer, load_aime_records
from opqad.evaluation.eval_vllm_turboquant import _chat_prompt, _request_stats, _write_report

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["eval", "bench", "smoke"], default="eval")
    ap.add_argument("--kv-cache-dtype", choices=["kvarn_k4v2_g128", "auto"], default="kvarn_k4v2_g128")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    assert 0 <= args.shard < args.num_shards

    import torch
    import vllm
    from vllm import SamplingParams

    assert torch.cuda.device_count() == 1, "Each TP=1 shard must see exactly one allocated GPU"
    fork = Path(vllm.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(fork), "rev-parse", "HEAD"], text=True).strip()
    assert revision == KVARN_REVISION, (fork, revision)
    llm, audit = make_engine(args)
    tokenizer = llm.get_tokenizer()
    records = load_aime_records(args.data, 30)
    assert len(records) == 30
    prompts = [_chat_prompt(tokenizer, f"{DEFAULT_INSTRUCTION}\n\n{r['problem']}") for r in records]
    prompt_hash = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
    token_hash = hashlib.sha256(json.dumps([
        tokenizer.encode(p, add_special_tokens=False) for p in prompts]).encode()).hexdigest()
    # Verified against the existing vLLM 0.21 / Transformers 4.57.6 environment.
    if "Nemotron-Research-Reasoning-Qwen-1.5B-brorl" in args.model:
        assert prompt_hash == "1c6cdbc22d8a1fa0bca4c4e09b3dbca04fc2d236503798f7b5fb504c5ec4e9b9"
        assert token_hash == "7c20c26c97887805d0d3ae24f350f19b17347cb233ebe5cbee53b6160eb0ffe5"
    base = {
        "evaluator_schema_version": 2, "evaluator": "qat_kvarn_packed",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_path_real": os.path.realpath(args.model),
        "benchmark": "aime_math", "data": str(args.data.resolve()), "subset": None,
        "dtype": "bfloat16", "kv_cache_dtype_requested": args.kv_cache_dtype,
        "tensor_parallel_size": 1, "max_model_len": args.max_model_len,
        "vllm_version": vllm.__version__, "fork_revision": revision,
        "fork_path": str(fork), "live_worker_audit": audit,
        "eager": args.eager, "max_num_seqs": args.max_num_seqs,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "prompt_sha256": prompt_hash, "prompt_token_ids_sha256": token_hash,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ["opqad/integrations/kvarn_vllm.py", "opqad/evaluation/eval_aime.py",
                                    "opqad/evaluation/eval_vllm_turboquant.py"]},
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ["vllm", "torch", "triton", "transformers", "flashinfer-python"]},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kvarn_environment": {k: v for k, v in os.environ.items() if k.startswith("KVARN_")},
    }
    print("LIVE_AUDIT " + json.dumps(audit), flush=True)

    if args.mode == "bench":
        # Identical token workloads: no EOS/length confounding. Warm each shape.
        rows = []
        ids = tokenizer.encode(prompts[0], add_special_tokens=False)
        for context in [2048, 8192, 24576]:
            batch = [{"prompt_token_ids": (ids * (context // len(ids) + 1))[:context]} for _ in range(32)]
            params = SamplingParams(temperature=0, max_tokens=256, min_tokens=256, ignore_eos=True)
            llm.generate(batch, params, use_tqdm=False)
            for repeat in range(3):
                torch.cuda.synchronize()
                t = time.perf_counter()
                outputs = llm.generate(batch, params, use_tqdm=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                assert tokens == 32 * 256
                rows.append({"input_tokens": context, "output_tokens_per_request": 256,
                             "batch": 32, "repeat": repeat, "elapsed_seconds": elapsed,
                             "output_tokens_per_second": tokens / elapsed})
                print("BENCH " + json.dumps(rows[-1]), flush=True)
                _write_report(args.output, {**base, "mode": "bench", "measurements": rows,
                                           "post_run_audit": llm.collective_rpc("kvarn_audit")})
        return

    keep = list(range(args.shard, len(records), args.num_shards))
    if args.mode == "smoke":
        keep = keep[:2]
    params = SamplingParams(n=args.n if args.mode == "eval" else 2,
                            temperature=0.7, top_p=0.95,
                            max_tokens=32768 if args.mode == "eval" else 512,
                            min_tokens=0 if args.mode == "eval" else 512,
                            stop_token_ids=[151645, 151643], seed=42)
    _write_report(args.output.with_suffix(".manifest.json"), {
        **base, "problem_indices": keep, "sampling": str(params), "status": "generating"})
    started = time.perf_counter()
    outputs = llm.generate([prompts[i] for i in keep], params, use_tqdm=True)
    elapsed = time.perf_counter() - started
    predictions = []
    for index, request in zip(keep, outputs):
        samples = []
        for j, output in enumerate(request.outputs):
            pred = extract_answer(output.text)
            samples.append({"sample": j, "prediction": pred,
                            "correct": answers_equal(pred, str(records[index]["answer"])),
                            "response": output.text, "token_ids": list(output.token_ids),
                            "stop_reason": output.stop_reason, **_request_stats(request, output)})
        predictions.append({"index": index, "id": records[index].get("id"),
                            "answer": str(records[index]["answer"]), "samples": samples,
                            "correct_samples": sum(s["correct"] for s in samples)})
    samples = [s for r in predictions for s in r["samples"]]
    count = len(samples)
    assert count == len(keep) * params.n
    lengths = [s["response_tokens"] for s in samples]
    correct = sum(s["correct"] for s in samples)
    truncated = sum(s["truncated"] for s in samples)
    eos = sum(s["finish_reason"] == "stop" for s in samples)
    report = {**base, "mode": args.mode, "thinking_enabled": True,
              "prompt_token_limit": None, "prompt_truncation": "none",
              "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": -1,
                           "n": params.n, "max_tokens": params.max_tokens,
                           "stop_token_ids": [151645, 151643], "seed": 42},
              "count": len(keep), "total_samples": count, "correct": correct,
              "accuracy": correct / count, "truncated_samples": truncated,
              "truncated_fraction": truncated / count, "eos_samples": eos,
              "eos_fraction": eos / count,
              "extraction_failures": sum(s["prediction"] is None for s in samples),
              "mean_response_tokens": statistics.mean(lengths),
              "median_response_tokens": statistics.median(lengths),
              "max_response_tokens": max(lengths),
              "mean_prompt_tokens": statistics.mean(s["prompt_tokens"] for s in samples),
              "max_prompt_tokens": max(s["prompt_tokens"] for s in samples),
              "elapsed_seconds": elapsed, "output_tokens_per_second": sum(lengths) / elapsed,
              "shard": args.shard, "num_shards": args.num_shards,
              "predictions": predictions}
    _write_report(args.output, report)
    print("RESULT " + json.dumps({k: v for k, v in report.items() if k not in {"predictions", "live_worker_audit"}}), flush=True)


if __name__ == "__main__":
    main()
