#!/usr/bin/env python3
"""Attribute the QAT step cost between the transformer body and the vocab head.

Job 22947 runs its reward stages 10-19x above the fwd+bwd compute floor measured
in job 22901.  That floor was measured on the transformer body alone; the actual
stages also materialize a full-vocabulary logits tensor and reduce it to a
top-K selection.  At this fork's shapes that tensor is

    LP_MB x 16896 x 151936 x 2 bytes  =  20.5 GB  for LP_MB=4

which is far more memory traffic than the body itself.  This measures the three
layers separately so the attribution is evidence rather than arithmetic.

Stages, each timed on its own:
  body      transformer blocks only, no lm_head
  +head     body + lm_head  (materializes [tokens, vocab])
  +topk     body + lm_head + chunked FP32 log-softmax gather over K+1 ids
             (what compute_log_prob / compute_distillation_reward actually do)
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def timed(fn, iters: int = 3):
    fn()  # warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (
        (time.perf_counter() - t0) / iters,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wa-state", default="")
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=16896)
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--backward", action="store_true", help="also time fwd+bwd")
    args = ap.parse_args()

    sys.path.insert(0, "/historical/home/QAT")
    sys.path.insert(0, "/historical/home/QAT/vendor/verl")
    from transformers import AutoModelForCausalLM
    from verl.utils.torch_functional import logprobs_for_ids_fp32

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable()
    model.train()

    label = "bf16"
    if args.wa_state:
        import modelopt.torch.opt as mto

        blob = torch.load(args.wa_state, map_location="cpu", weights_only=False)
        mto.restore_from_modelopt_state(model, modelopt_state=blob["modelopt_state"])
        model.load_state_dict(blob.get("quantizer_buffers") or {}, strict=False)
        model.train()
        label = "nvfp4"

    B, T, K = args.bsz, args.seqlen, args.top_k
    vocab = model.config.vocab_size
    ids = torch.randint(0, 1000, (B, T), device="cuda")
    topk_ids = torch.randint(0, vocab, (B, T, K), device="cuda")
    logits_gb = B * T * vocab * 2 / 2**30
    print(
        f"[profile] {label}  B={B} T={T} K={K} vocab={vocab}\n"
        f"[profile] a full BF16 logits tensor for this shape is {logits_gb:.1f} GB",
        flush=True,
    )

    body = model.model  # transformer blocks, no lm_head

    def f_body():
        with torch.no_grad():
            body(input_ids=ids)

    def f_head():
        with torch.no_grad():
            h = body(input_ids=ids)[0]
            model.lm_head(h)

    def f_topk():
        with torch.no_grad():
            h = body(input_ids=ids)[0]
            logits = model.lm_head(h)
            sel = torch.cat((ids.unsqueeze(-1), topk_ids), dim=-1)
            logprobs_for_ids_fp32(logits, sel, temperature=1.0)

    rows = []
    for name, fn in (("body", f_body), ("+head", f_head), ("+topk", f_topk)):
        try:
            sec, gb = timed(fn, args.iters)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {name:<6} OOM", flush=True)
            continue
        rows.append((name, sec, gb))
        print(f"  {name:<6} {sec:7.3f} s   peak={gb:6.1f} GB", flush=True)

    if len(rows) >= 2:
        print(
            f"\n[profile] lm_head alone: {rows[1][1] - rows[0][1]:.3f} s "
            f"({(rows[1][1] - rows[0][1]) / rows[1][1] * 100:.0f}% of +head)"
        )
    if len(rows) >= 3:
        print(
            f"[profile] top-k log-softmax alone: {rows[2][1] - rows[1][1]:.3f} s"
        )
        print(
            f"[profile] vocab-head total: {rows[2][1] - rows[0][1]:.3f} s "
            f"vs body {rows[0][1]:.3f} s  -> "
            f"{(rows[2][1] - rows[0][1]) / rows[0][1]:.1f}x the body"
        )

    if args.backward:
        def f_bwd():
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)

        try:
            sec, gb = timed(f_bwd, args.iters)
            print(f"\n  fwd+bwd (with head) {sec:7.3f} s   peak={gb:6.1f} GB")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("\n  fwd+bwd OOM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
