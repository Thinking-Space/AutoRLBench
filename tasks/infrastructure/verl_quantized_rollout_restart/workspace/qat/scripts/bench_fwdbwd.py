#!/usr/bin/env python3
"""Single-GPU fwd+bwd throughput sweep for the QAT actor.

Answers one question: at what micro-batch / sequence length does the 1.5B
actor actually saturate the card, and what does each fake-quant scheme cost?

This deliberately runs WITHOUT FSDP.  The number it produces is the compute
ceiling for one rank; the gap between it and the observed per-micro-batch time
in a real 8-GPU step is the FSDP communication + verl overhead.

Variants:
  bf16    plain model
  kvq     + TurboQuant KV fake-quant (the scheme we train today)
  nvfp4   + modelopt NVFP4 W4A4 fake-quant on every nn.Linear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def build_model(model_path: str, variant: str, kv_scheme: str, grad_ckpt: bool,
                attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    # Must match what verl actually runs.  flash_attn is not installed in this
    # environment, so preflight pins the actor and the reward model to sdpa
    # (+actor_rollout_ref.model.override_config.attn_implementation=sdpa).
    # Benchmarking FA2 would measure a configuration training never uses.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attn
    ).cuda()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    n_quant = 0
    if variant == "kvq":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from qat_kv.turboquant_attn_hook import install_turboquant_fake_quant

        install_turboquant_fake_quant(model, kv_scheme)
        n_quant = -1  # installed as attention hooks, not TensorQuantizer modules
    elif variant == "nvfp4":
        import modelopt.torch.quantization as mtq

        cfg = getattr(mtq, "NVFP4_DEFAULT_CFG")

        def forward_loop(m):
            # one tiny calibration pass just to populate amax
            ids = torch.randint(0, 1000, (1, 512), device="cuda")
            m(input_ids=ids)

        mtq.quantize(model, cfg, forward_loop=forward_loop)
        from modelopt.torch.quantization.nn import TensorQuantizer

        n_quant = sum(1 for m in model.modules() if isinstance(m, TensorQuantizer))
        model.train()
    return model, n_quant


def timed(model, bsz: int, seqlen: int, iters: int) -> dict:
    ids = torch.randint(0, 1000, (bsz, seqlen), device="cuda")
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()

    def one():
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    one()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tokens = bsz * seqlen
    return {
        "bsz": bsz,
        "seqlen": seqlen,
        "tokens": tokens,
        "sec": round(dt, 4),
        "tok_per_s": round(tokens / dt, 1),
        # fwd+bwd on a dense transformer ~= 6 * params * tokens
        "tflops": round(6 * 1.54e9 * tokens / dt / 1e12, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default="bf16,kvq,nvfp4")
    ap.add_argument("--kv-scheme", default="turboquant_k2v3_nc")
    ap.add_argument("--grid", default="1x2048,1x8192,2x8192,4x8192,8x8192,2x16384")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation (verl uses sdpa)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    grid = []
    for item in args.grid.split(","):
        b, _, s = item.strip().partition("x")
        grid.append((int(b), int(s)))

    rows = []
    for variant in args.variants.split(","):
        variant = variant.strip()
        print(f"\n=== variant={variant} attn={args.attn} ===", flush=True)
        try:
            model, n_quant = build_model(
                args.model, variant, args.kv_scheme, not args.no_grad_ckpt, args.attn
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {variant}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"[info] {variant}: quantizers={n_quant}", flush=True)
        for bsz, seqlen in grid:
            try:
                r = timed(model, bsz, seqlen, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {bsz}x{seqlen}: OOM", flush=True)
                continue
            r["variant"] = variant
            rows.append(r)
            print(
                f"  {bsz:>2}x{seqlen:<6} tokens={r['tokens']:>6} "
                f"{r['sec']:>7.3f}s  {r['tok_per_s']:>9.1f} tok/s  "
                f"{r['tflops']:>6.1f} TFLOP/s  peak={r['peak_gb']}GB",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\n[wrote] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
