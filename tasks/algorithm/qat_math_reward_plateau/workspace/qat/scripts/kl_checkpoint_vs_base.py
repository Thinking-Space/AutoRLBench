#!/usr/bin/env python3
"""Exact full-vocabulary KL of a KV3 checkpoint against the base-FP policy.

Both policies are teacher-forced on the same fixed base-FP trajectory.  The
candidate uses the repository's deployment-matched TurboQuant k3v4 fake
quantizer with the vLLM prefill/decode split; the base model is unquantized.
All probability arithmetic is FP32 and KL values are reported in nats/token.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qat_kv.turboquant_attn_hook import install_turboquant_fake_quant


def _load_model(path: str, dtype: torch.dtype):
    kwargs = dict(trust_remote_code=True, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)
    return model.to("cuda").eval()


def _backbone(model):
    for name in ("model", "transformer"):
        value = getattr(model, name, None)
        if value is not None:
            return value
    raise TypeError(f"cannot find causal-LM backbone on {type(model).__name__}")


def _last_hidden(model, input_ids: torch.Tensor, prompt_len: int, quant_state=None):
    context = quant_state.prefill_exact(prompt_len) if quant_state is not None else nullcontext()
    with context, torch.inference_mode():
        output = _backbone(model)(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
    hidden = output.last_hidden_state
    # hidden[t-1] predicts token[t].  Score every response token, but no prompt
    # token and no position after the final response token.
    response_len = input_ids.shape[1] - prompt_len
    scored = hidden[:, prompt_len - 1 : prompt_len - 1 + response_len, :].contiguous()
    del output, hidden
    return scored


class Running:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        self.count += values.numel()
        self.total += float(values.sum().item())
        self.total_sq += float((values * values).sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))

    def report(self) -> dict[str, float | int]:
        mean = self.total / max(1, self.count)
        variance = max(0.0, self.total_sq / max(1, self.count) - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum if self.count else math.nan,
            "max": self.maximum if self.count else math.nan,
        }


def _summarize(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = statistics.fmean(values) if values else math.nan
    std = statistics.stdev(values) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(n) if n else math.nan,
        "ci95_normal_low": mean - 1.96 * std / math.sqrt(n) if n else math.nan,
        "ci95_normal_high": mean + 1.96 * std / math.sqrt(n) if n else math.nan,
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def _score_sample(base, candidate, state, row: dict[str, Any], logit_chunk: int):
    prompt_ids = [int(x) for x in row["prompt_token_ids"]]
    response_ids = [int(x) for x in row["response_token_ids"]]
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
    prompt_len = len(prompt_ids)

    state.reset_counters()
    base_hidden = _last_hidden(base, ids, prompt_len)
    candidate_hidden = _last_hidden(candidate, ids, prompt_len, state)
    state.assert_layers_quantized()

    metrics = {
        name: Running()
        for name in (
            "kl_candidate_to_base",
            "kl_base_to_candidate",
            "js_divergence",
            "base_entropy",
            "candidate_entropy",
            "total_variation",
            "top1_agreement",
        )
    }
    position_kl = [Running() for _ in range(10)]
    token_count = len(response_ids)
    for start in range(0, token_count, logit_chunk):
        stop = min(start + logit_chunk, token_count)
        with torch.inference_mode():
            base_logits = base.lm_head(base_hidden[:, start:stop]).float()
            cand_logits = candidate.lm_head(candidate_hidden[:, start:stop]).float()
            base_logp = torch.log_softmax(base_logits, dim=-1)
            cand_logp = torch.log_softmax(cand_logits, dim=-1)
            base_p = base_logp.exp()
            cand_p = cand_logp.exp()
            rev = (cand_p * (cand_logp - base_logp)).sum(dim=-1).squeeze(0)
            fwd = (base_p * (base_logp - cand_logp)).sum(dim=-1).squeeze(0)
            midpoint = 0.5 * (base_p + cand_p)
            log_midpoint = midpoint.log()
            js = 0.5 * (
                (base_p * (base_logp - log_midpoint)).sum(dim=-1)
                + (cand_p * (cand_logp - log_midpoint)).sum(dim=-1)
            ).squeeze(0)
            base_entropy = -(base_p * base_logp).sum(dim=-1).squeeze(0)
            cand_entropy = -(cand_p * cand_logp).sum(dim=-1).squeeze(0)
            tv = (0.5 * (base_p - cand_p).abs().sum(dim=-1)).squeeze(0)
            top1 = (base_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).float().squeeze(0)

        chunk_values = {
            "kl_candidate_to_base": rev,
            "kl_base_to_candidate": fwd,
            "js_divergence": js,
            "base_entropy": base_entropy,
            "candidate_entropy": cand_entropy,
            "total_variation": tv,
            "top1_agreement": top1,
        }
        for name, values in chunk_values.items():
            metrics[name].add(values)
        for offset, value in enumerate(rev):
            position_bin = min(9, int((start + offset) * 10 / max(1, token_count)))
            position_kl[position_bin].add(value.reshape(1))

        del base_logits, cand_logits, base_logp, cand_logp, base_p, cand_p
        del midpoint, log_midpoint, rev, fwd, js, base_entropy, cand_entropy, tv, top1

    del ids, base_hidden, candidate_hidden
    torch.cuda.empty_cache()
    return {
        "split": row["split"],
        "stratum": row["stratum"],
        "benchmark": row["benchmark"],
        "example_id": row["example_id"],
        "difficulty": row.get("difficulty"),
        "prompt_tokens": prompt_len,
        "response_tokens": token_count,
        "metrics": {name: meter.report() for name, meter in metrics.items()},
        "reverse_kl_by_relative_position_decile": [meter.report() for meter in position_kl],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(rows[0]["metrics"])
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "token_count": sum(row["response_tokens"] for row in rows),
        "prompt_tokens_mean": statistics.fmean(row["prompt_tokens"] for row in rows),
        "response_tokens_mean": statistics.fmean(row["response_tokens"] for row in rows),
        "stratum_counts": {},
        "metrics": {},
    }
    for row in rows:
        key = row["stratum"]
        result["stratum_counts"][key] = result["stratum_counts"].get(key, 0) + 1
    for name in metric_names:
        sample_means = [row["metrics"][name]["mean"] for row in rows]
        token_total = sum(
            row["metrics"][name]["mean"] * row["metrics"][name]["count"] for row in rows
        )
        token_count = sum(row["metrics"][name]["count"] for row in rows)
        result["metrics"][name] = {
            "sample_macro": _summarize(sample_means),
            "token_micro_mean": token_total / max(1, token_count),
        }
    deciles = []
    for index in range(10):
        count = sum(row["reverse_kl_by_relative_position_decile"][index]["count"] for row in rows)
        total = sum(
            row["reverse_kl_by_relative_position_decile"][index]["mean"]
            * row["reverse_kl_by_relative_position_decile"][index]["count"]
            for row in rows
        )
        deciles.append({"decile": index + 1, "count": count, "token_micro_mean": total / max(1, count)})
    result["reverse_kl_by_relative_position_decile"] = deciles
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-scheme", default="turboquant_k3v4_nc")
    parser.add_argument("--logit-chunk", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.logit_chunk < 1:
        raise ValueError("--logit-chunk must be positive")

    sample_manifest = json.loads(args.samples.read_text())
    rows = sample_manifest["samples"]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("sample manifest is empty")

    started = time.time()
    dtype = torch.bfloat16
    print(f"[kl] loading base: {args.base}", flush=True)
    base = _load_model(args.base, dtype)
    print(f"[kl] loading candidate: {args.candidate}", flush=True)
    candidate = _load_model(args.candidate, dtype)
    state = install_turboquant_fake_quant(candidate, args.kv_scheme)

    results = []
    for index, row in enumerate(rows, start=1):
        t0 = time.time()
        result = _score_sample(base, candidate, state, row, args.logit_chunk)
        results.append(result)
        print(
            f"[kl] {index}/{len(rows)} {row['split']} {row['stratum']} "
            f"tokens={result['response_tokens']} "
            f"revKL={result['metrics']['kl_candidate_to_base']['mean']:.6g} "
            f"seconds={time.time()-t0:.1f}",
            flush=True,
        )

    by_split = {
        split: _aggregate([row for row in results if row["split"] == split])
        for split in ("train", "test")
        if any(row["split"] == split for row in results)
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_model": args.base,
        "candidate_model": args.candidate,
        "candidate_kv_scheme": args.kv_scheme,
        "base_kv": "fp_bfloat16",
        "candidate_kv": "deployment-matched TurboQuant fake-dequant, prefill_exact",
        "dtype": "bfloat16 weights/hidden; float32 softmax and KL",
        "trajectory_conditioning": "teacher forcing on the same fixed base-FP continuation",
        "primary_metric": "KL(candidate_KV3 || base_FP), sample-macro mean, nats/token",
        "sample_manifest": str(args.samples),
        "sample_manifest_selection": sample_manifest.get("selection"),
        "quantizer": state.metadata(),
        "elapsed_seconds": time.time() - started,
        "aggregates": by_split,
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[kl] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
