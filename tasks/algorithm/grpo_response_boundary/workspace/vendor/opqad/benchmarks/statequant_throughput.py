"""Measure whole-model StateQuant throughput across concurrency levels."""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import os
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Print the protocol without importing vLLM or starting a GPU")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=["stock-fp32", "stock-bf16", "pace4", "int4v", "bf16"])
    p.add_argument("--modes", default="stock-fp32,stock-bf16,pace4,int4v")
    p.add_argument("--batches", default="1,16,32,64,128,256,512")
    p.add_argument("--input-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--performance-mode", choices=["balanced", "throughput"], default="throughput")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--cuda-graph-sizes", help="Comma-separated native CUDA Graph capture sizes")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--buffer-bits", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    config, _ = p.parse_known_args()
    if config.config:
        defaults = json.loads(config.config.read_text())
        allowed = {a.dest for a in p._actions}
        unknown = set(defaults) - allowed
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**defaults)
    return p


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def orchestrate(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), root / "opqad/integrations/statequant.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    save(args.output / "manifest.json", {
        "command": [sys.executable, "-m", "opqad.benchmarks.statequant_throughput", *sys.argv[1:]],
        "git_revision": revision.stdout.strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })
    statuses = []
    monitor_cmd = ["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw",
                   "--format=csv", "--loop-ms=1000"]
    device = os.environ.get("SLURM_JOB_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if device:
        monitor_cmd += ["--id=" + device]
    with (args.output / "gpu.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(monitor_cmd, stdout=gpu_log, stderr=subprocess.STDOUT)
        try:
            for mode in args.modes.split(","):
                log = args.output / f"{mode}.log"
                cmd = [sys.executable, "-m", "opqad.benchmarks.statequant_throughput", *sys.argv[1:], "--mode", mode]
                print(f"START {mode}: {log}", flush=True)
                with log.open("w") as stream:
                    result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
                statuses.append({"mode": mode, "exit_code": result.returncode, "log": str(log)})
                save(args.output / "status.json", statuses)
                print(f"DONE {mode}: exit {result.returncode}", flush=True)
                if result.returncode:
                    break
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    summaries = []
    for mode in args.modes.split(","):
        path = args.output / f"{mode}.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    save(args.output / "summary.json", summaries)
    return int(any(s["exit_code"] for s in statuses))


def measure(args) -> int:
    from opqad.integrations.statequant import configure_environment, verify_cache_reports

    env = configure_environment(args.mode, window=args.window, buffer_bits=args.buffer_bits)
    from vllm import LLM, SamplingParams

    model_config = json.loads((Path(args.model) / "config.json").read_text())
    expected_layers = model_config["hybrid_override_pattern"].count("M")
    dtype = "bfloat16" if args.mode == "stock-bf16" else "float32"
    settings = {
        "model": args.model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": False,
        "use_replayssm": False,
        "mamba_cache_mode": "none",
        "mamba_ssm_cache_dtype": dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "performance_mode": args.performance_mode,
        "worker_extension_cls": "opqad.integrations.statequant.StateQuantWorkerExtension",
        # V2 synthetic all-short-prefill warmup has O(max_num_seqs * state) FP32 scratch.
        # Warm the actual fixed-prefix workload below instead.
        "kernel_config": {"enable_jit_warmup": False},
    }
    if args.cuda_graph_sizes:
        sizes = sorted(set(map(int, args.cuda_graph_sizes.split(","))))
        settings["compilation_config"] = {"cudagraph_capture_sizes": sizes, "max_cudagraph_capture_size": max(sizes)}
    result = {
        "mode": args.mode,
        "engine_settings": settings,
        "environment": env,
        "protocol": {"input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                     "repeats": args.repeats, "seed": args.seed,
                     "metric": "sum generated output tokens / elapsed wall seconds, including prefill",
                     "aggregation": "median of repeated whole-batch output-token rates"},
        "versions": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers", "statequant", "triton")},
        "cases": [],
    }
    model_source = Path(args.model) / "opqad_source.json"
    if model_source.exists():
        result["model_source"] = json.loads(model_source.read_text())
    output = args.output / f"{args.mode}.json"
    save(output, result)
    init_start = time.perf_counter()
    llm = LLM(**settings)
    result["initialization_seconds"] = time.perf_counter() - init_start
    reports = llm.collective_rpc("statequant_cache_report")
    verify_cache_reports(reports, args.mode, expected_layers)
    result["worker_cache_reports"] = reports
    import torch

    result["gpu"] = {"name": torch.cuda.get_device_name(0), "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    save(output, result)
    rng = random.Random(args.seed)
    # Token IDs avoid tokenizer cost and accidental shared-prefix reuse. Every
    # format receives identical token sequences; output length is forced.
    prompts = [{"prompt_token_ids": [rng.randrange(100, min(model_config["vocab_size"], 32000))
                                    for _ in range(args.input_tokens)]}
               for _ in range(max(map(int, args.batches.split(","))))]
    sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                              min_tokens=args.output_tokens, ignore_eos=True, detokenize=False)
    warmup = SamplingParams(temperature=0, max_tokens=args.warmup_tokens,
                            ignore_eos=True, detokenize=False)
    # Prime the full context-length range once, outside measurement.
    llm.generate(prompts[:1], sampling, use_tqdm=False)
    for batch in map(int, args.batches.split(",")):
        llm.generate(prompts[:batch], warmup, use_tqdm=False)
        repetitions = []
        for repeat in range(args.repeats):
            started_at_unix = time.time()
            begin = time.perf_counter()
            outputs = llm.generate(prompts[:batch], sampling, use_tqdm=False)
            elapsed = time.perf_counter() - begin
            lengths = [len(o.outputs[0].token_ids) for o in outputs]
            if len(lengths) != batch or any(n != args.output_tokens for n in lengths):
                raise RuntimeError(f"Incomplete generation at batch {batch}: {lengths}")
            repetitions.append({"repeat": repeat, "elapsed_seconds": elapsed,
                                "requests": batch, "output_tokens": sum(lengths),
                                "output_tok_s": sum(lengths) / elapsed,
                                "started_at_unix": started_at_unix, "finished_at_unix": time.time()})
        case = {"batch_size": batch, "engine_concurrency_cap": args.max_num_seqs,
                "median_output_tok_s": statistics.median(r["output_tok_s"] for r in repetitions),
                "repetitions": repetitions}
        result["cases"].append(case)
        result["peak"] = max(result["cases"], key=lambda c: c["median_output_tok_s"])
        save(output, result)
        print(json.dumps({"mode": args.mode, **case}), flush=True)
    del llm
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.input_tokens + args.output_tokens > args.max_model_len:
        raise SystemExit("Input + output exceeds max-model-len")
    if args.repeats < 1 or args.warmup_tokens < args.window:
        raise SystemExit("Need a repetition and warmup long enough to fold the state buffer")
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    return measure(args) if args.mode else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
