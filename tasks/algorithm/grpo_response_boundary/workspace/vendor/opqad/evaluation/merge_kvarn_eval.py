#!/usr/bin/env python3
"""Validate complete KVarN shards, reuse QAT scoring merge, fix derived metrics."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def validate(shards):
    assert len(shards) == 2, "Expected exactly two shards"
    assert {s["shard"] for s in shards} == {0, 1}
    identity = ["fork_revision", "data_sha256", "prompt_sha256", "evaluator_sha256",
                "sampling", "eager", "max_num_seqs", "kvarn_environment",
                "source_sha256", "package_versions"]
    for s in shards:
        assert s["mode"] == "eval" and s["num_shards"] == 2
        assert s["sampling"]["n"] == 16
        assert len(s["predictions"]) == 15
        assert all(len(r["samples"]) == 16 for r in s["predictions"])
        assert all(s[k] == shards[0][k] for k in identity), "Shard settings differ"
    indices = [r["index"] for s in shards for r in s["predictions"]]
    assert sorted(indices) == list(range(30)), "Missing or repeated AIME problems"


def write_summary(path, report):
    """Compare paired workloads on each GPU; never mix cold AIME timing in."""
    lines = ["# KVarN K4V2 / broRL 1.5B", "",
             f"AIME24 avg@16: **{report['accuracy']:.4f}** "
             f"({report['correct']}/{report['total_samples']}).",
             f"Problem-cluster SE: {report['problem_cluster_standard_error']:.4f}.", "",
             f"EOS: {report['eos_fraction']:.2%}; truncated: {report['truncated_fraction']:.2%}.",
             f"Mean output: {report['mean_response_tokens']:.1f} tokens; "
             f"median: {report['median_response_tokens']:.1f}.", "",
             "## Paired warmed throughput", "",
             "Batch=32, output=256 tokens, ignore EOS, median of three runs per GPU.",
             "Times include prefill and decode. These are not decode-only timings.", "",
             "| Shard | Input tokens | BF16 tok/s | KVarN tok/s | KVarN / BF16 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for shard in range(2):
        files = [path.parent / f"bench-{kv}-shard{shard}.json"
                 for kv in ["auto", "kvarn_k4v2_g128"]]
        if not all(p.exists() for p in files):
            lines.append(f"\nShard {shard} benchmark files unavailable.")
            continue
        baseline, quantized = [json.loads(p.read_text()) for p in files]
        for field in ["model", "dtype", "fork_revision", "max_num_seqs", "eager"]:
            assert baseline[field] == quantized[field], f"Unmatched speed test: {field}"
        for field in ["gpu", "compute_capability", "gpu_total_bytes"]:
            assert baseline["live_worker_audit"][0][field] == quantized["live_worker_audit"][0][field]
        for context in [2048, 8192, 24576]:
            speeds = []
            for run in [baseline, quantized]:
                rows = [m for m in run["measurements"] if m["input_tokens"] == context]
                assert len(rows) == 3
                assert all(m["batch"] == 32 and m["output_tokens_per_request"] == 256 for m in rows)
                speeds.append(statistics.median(m["output_tokens_per_second"] for m in rows))
            lines.append(f"| {shard} | {context} | {speeds[0]:.1f} | {speeds[1]:.1f} | {speeds[1]/speeds[0]:.3f}x |")
    lines.extend(["", "Full trajectories and provenance: `merged.json`; "
                  "per-shape measurements and GPU audit: `bench-*.json`.", ""])
    path.with_name("summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path, nargs=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shards = [json.loads(p.read_text()) for p in args.shards]
    validate(shards)
    subprocess.run([sys.executable, str(Path(__file__).with_name("merge_eval_shards.py")),
                    "--out", str(args.out), *map(str, args.shards)], check=True)
    report = json.loads(args.out.read_text())
    samples = [s for r in report["predictions"] for s in r["samples"]]
    report["median_response_tokens"] = statistics.median(s["response_tokens"] for s in samples)
    report["eos_samples"] = sum(s["finish_reason"] == "stop" for s in samples)
    report["eos_fraction"] = report["eos_samples"] / len(samples)
    report.pop("output_tokens_per_second", None)
    report["output_tokens_per_gpu_second"] = sum(s["response_tokens"] for s in samples) / report["elapsed_seconds_sum"]
    report["shard_timing"] = [{"shard": s["shard"], "elapsed_seconds": s["elapsed_seconds"],
                               "output_tokens_per_second": s["output_tokens_per_second"]} for s in shards]
    report["live_worker_audit"] = [w for s in shards for w in s["live_worker_audit"]]
    # Empirical uncertainty clusters all 16 attempts of the same problem.
    rates = [r["correct_samples"] / 16 for r in report["predictions"]]
    report["problem_cluster_standard_error"] = statistics.stdev(rates) / len(rates) ** 0.5
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_summary(args.out, report)


if __name__ == "__main__":
    main()
