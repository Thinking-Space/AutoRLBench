#!/usr/bin/env python3
"""Why W4A4 collapses: compare logits and per-layer activations against BF16.

The eval JSON only holds generated text, so the divergence has to be measured
by re-running forwards.  All three variants are built from the same BF16
weights with modelopt's simulated quantization, so the only difference is which
tensors get passed through a quantizer:

    bf16    nothing quantized                (reference)
    w4a16   weights only                     (Marlin serves this at inference)
    w4a4    weights and layer inputs         (cutlass/flashinfer serves this)

Two things are measured on identical inputs:

  * final logits -- KL(bf16 || variant) per token plus top-1 agreement, which is
    what actually decides the sampled token
  * per-layer hidden states -- relative error, to locate where the error is
    introduced rather than only observing it at the output

The PTQ summary already hints at the mechanism: mlp.down_proj.input_quantizer
calibrated to amax 4.72e+02 while gate_proj.input_quantizer sits at 4.55e+01.
E2M1 spans about +-6 with one FP8 scale per 16 values, so a single outlier in a
block sets the scale for the other fifteen.  Per-layer input ranges are printed
so that hypothesis can be checked against where the error actually appears.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_inputs(args, tokenizer):
    import pyarrow.parquet as pq

    cols = pq.read_table(args.data, columns=["prompt_token_ids", "response_token_ids"]).to_pydict()
    seqs = []
    for p, r in zip(cols["prompt_token_ids"], cols["response_token_ids"]):
        ids = (list(p) + list(r))[: args.seq_len]
        if len(ids) == args.seq_len:
            seqs.append(ids)
        if len(seqs) >= args.batch:
            break
    if not seqs:
        raise SystemExit("no sequences long enough for the requested seq_len")
    return torch.tensor(seqs, dtype=torch.long)


def build(model_path, cfg_name, calib_ids):
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    if cfg_name is None:
        return model

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, calib_ids.size(0), 2):
                m(input_ids=calib_ids[i : i + 2].cuda())

    cfg = resolve_cfg(cfg_name)
    return mtq.quantize(model, cfg, forward_loop=forward_loop)


def resolve_cfg(cfg_name):
    """Named config, optionally with down_proj input quantization switched off.

    The PTQ summary puts mlp.down_proj.input_quantizer at amax 242 mean / 3088
    max while every other input sits between 8 and 24.  E2M1 spans +-6 with one
    FP8 scale per 16 values, so one outlier in a block flattens the other
    fifteen.  These ablations keep everything else identical and only stop
    quantizing that one input, which is what isolates the hypothesis -- the
    weights stay NVFP4 throughout, so any recovery is attributable to
    activations alone.
    """
    import copy

    import modelopt.torch.quantization as mtq

    base, _, mode = cfg_name.partition("|")
    cfg = copy.deepcopy(getattr(mtq, base))
    if not mode:
        return cfg
    if mode == "nodp":
        pats = ["*down_proj.input_quantizer"]
    elif mode == "nodp_top3":
        # The three layers holding the largest input amax (2: 3088, 26: 1232,
        # 27: 472).  If these alone recover most of the gap the fix is nearly
        # free; if it takes all 28 layers the outliers are diffuse.
        pats = [f"*layers.{i}.mlp.down_proj.input_quantizer" for i in (2, 26, 27)]
    else:
        raise SystemExit(f"unknown ablation mode: {mode}")
    q = cfg["quant_cfg"]
    entries = [{"quantizer_name": p, "enable": False} for p in pats]
    if isinstance(q, list):
        q.extend(entries)
    else:  # older dict-shaped quant_cfg
        for p in pats:
            q[p] = {"enable": False}
    return cfg


@torch.no_grad()
def run(model, ids):
    out = model(input_ids=ids.cuda(), output_hidden_states=True)
    return out.logits.float().cpu(), [h.float().cpu() for h in out.hidden_states]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = load_inputs(args, tokenizer)
    print(f"[diag] inputs {tuple(ids.shape)}", flush=True)

    report = {"batch": args.batch, "seq_len": args.seq_len, "variants": {}}
    ref_logits = ref_hidden = None

    variants = (
        ("bf16", None),
        ("w4a16", "W4A16_NVFP4_CFG"),
        ("w4a4", "NVFP4_DEFAULT_CFG"),
        ("w4a4_nodp", "NVFP4_DEFAULT_CFG|nodp"),
        ("w4a4_nodp_top3", "NVFP4_DEFAULT_CFG|nodp_top3"),
    )
    for name, cfg in variants:
        print(f"\n=== {name} ===", flush=True)
        model = build(args.model, cfg, ids)

        if name == "w4a4":
            # Where does each layer's input dynamic range sit?  This is the
            # quantity the block scale has to cover.
            ranges = []
            for n, m in model.named_modules():
                q = getattr(m, "input_quantizer", None)
                amax = getattr(q, "amax", None) if q is not None else None
                if amax is not None and getattr(q, "is_enabled", True):
                    ranges.append((n, float(amax.max())))
            ranges.sort(key=lambda t: -t[1])
            report["input_amax_top"] = ranges[:8]
            report["input_amax_bottom"] = ranges[-5:]
            print("  largest input amax:", [(n.split('.')[-2] + '.' + n.split('.')[-1], f"{v:.1f}")
                                            for n, v in ranges[:5]], flush=True)

        logits, hidden = run(model, ids)

        if name == "bf16":
            ref_logits, ref_hidden = logits, hidden
        else:
            lp_ref = F.log_softmax(ref_logits, dim=-1)
            lp_var = F.log_softmax(logits, dim=-1)
            kl = (lp_ref.exp() * (lp_ref - lp_var)).sum(-1)
            top1 = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean()
            layer_rel = [
                float((h - r).norm() / r.norm().clamp(min=1e-9))
                for r, h in zip(ref_hidden, hidden)
            ]
            entry = {
                "kl_mean_nat": float(kl.mean()),
                "kl_p99_nat": float(kl.flatten().kthvalue(int(0.99 * kl.numel()))[0]),
                "kl_max_nat": float(kl.max()),
                "top1_agreement": float(top1),
                "logit_max_abs_diff": float((logits - ref_logits).abs().max()),
                "hidden_rel_err_by_layer": [round(v, 5) for v in layer_rel],
            }
            report["variants"][name] = entry
            print(f"  KL(bf16||{name}) mean={entry['kl_mean_nat']:.4f} "
                  f"p99={entry['kl_p99_nat']:.4f} max={entry['kl_max_nat']:.3f} nat", flush=True)
            print(f"  top-1 agreement: {entry['top1_agreement'] * 100:.2f}%", flush=True)
            print(f"  hidden rel err: layer0={layer_rel[0]:.4f} "
                  f"mid={layer_rel[len(layer_rel) // 2]:.4f} last={layer_rel[-1]:.4f}", flush=True)

        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
