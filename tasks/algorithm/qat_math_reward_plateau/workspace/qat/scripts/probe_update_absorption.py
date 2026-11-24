#!/usr/bin/env python3
"""Does NVFP4 weight quantization swallow the optimizer update?

QuRL reports that RL weight updates (1e-7..1e-6) are far smaller than INT8
quantization error, so "quantization masks nearly all weight updates,
effectively freezing the quantized model".  Our lr is 1.5e-6, squarely in that
range, so the same question decides whether NVFP4 weight QAT can move anything
at all.

What is measured, for a real gradient and a real AdamW step:

  * the fraction of weight elements whose *quantised* value changes
  * how much of the intended update survives quantisation,
    ||Q(theta1) - Q(theta0)|| / ||theta1 - theta0||

The first AdamW step is worth stating explicitly because it makes the scale
unambiguous: with bias correction, m_hat / sqrt(v_hat) = g / |g| = sign(g), so
step one displaces every weight by almost exactly lr, independent of gradient
magnitude.  A learning-rate sweep therefore maps directly onto displacement.

Both a single step and the cumulative drift of a full run are reported: master
weights are kept in BF16 and accumulate, so a per-step verdict of "frozen" does
not by itself mean 150 steps are frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def quantized_view(model):
    """Q(W) for every Linear that modelopt gave a weight quantizer."""
    out = {}
    for name, mod in model.named_modules():
        q = getattr(mod, "weight_quantizer", None)
        if q is None or not getattr(q, "is_enabled", True):
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        with torch.no_grad():
            out[name] = q(w.detach()).float().cpu()
    return out


def block_step_size(w, block=16):
    """Quantisation step inside each 16-value block, for E2M1 (max magnitude 6)."""
    flat = w.detach().float().flatten()
    n = (flat.numel() // block) * block
    blocks = flat[:n].view(-1, block)
    return (blocks.abs().amax(dim=1) / 6.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default="1.5e-6,1e-5,1e-4,1e-3")
    ap.add_argument("--steps-to-project", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    import modelopt.torch.quantization as mtq
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM

    cols = pq.read_table(args.data, columns=["prompt_token_ids", "response_token_ids"]).to_pydict()
    ids = None
    for p, r in zip(cols["prompt_token_ids"], cols["response_token_ids"]):
        seq = (list(p) + list(r))[: args.seq_len]
        if len(seq) == args.seq_len:
            ids = torch.tensor([seq], dtype=torch.long)
            break
    if ids is None:
        raise SystemExit("no sequence long enough")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    def forward_loop(m):
        with torch.no_grad():
            m(input_ids=ids.cuda())

    model = mtq.quantize(model, mtq.W4A16_NVFP4_CFG, forward_loop=forward_loop)
    model.train()

    report = {"lr_sweep": {}, "steps_projected": args.steps_to_project}

    # How coarse is the grid the weights have to cross?
    steps = []
    for name, mod in model.named_modules():
        if getattr(mod, "weight_quantizer", None) is not None and hasattr(mod, "weight"):
            steps.append(block_step_size(mod.weight))
    allsteps = torch.cat(steps)
    report["quant_step"] = {
        "median": float(allsteps.median()),
        "p10": float(allsteps.kthvalue(max(1, int(0.10 * allsteps.numel())))[0]),
        "p90": float(allsteps.kthvalue(max(1, int(0.90 * allsteps.numel())))[0]),
    }
    print(f"[probe] quantisation step per 16-value block: "
          f"p10={report['quant_step']['p10']:.5f} "
          f"median={report['quant_step']['median']:.5f} "
          f"p90={report['quant_step']['p90']:.5f}", flush=True)

    theta0 = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    q0 = quantized_view(model)

    # One real backward: the gradient's sign pattern decides which weights sit
    # near a rounding boundary, so a synthetic update would not answer this.
    out = model(input_ids=ids.cuda(), labels=ids.cuda())
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    print(f"[probe] loss={float(out.loss):.4f}  params with grad={len(grads)}", flush=True)

    for lr in [float(x) for x in args.lrs.split(",")]:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in grads:
                    # AdamW step 1 after bias correction: lr * g/(|g|+eps).
                    g = grads[n]
                    p.copy_(theta0[n] - lr * g / (g.abs() + 1e-8))
        q1 = quantized_view(model)

        changed = total = 0
        num = den = 0.0
        for n in q0:
            d = (q1[n] - q0[n])
            changed += int((d != 0).sum())
            total += d.numel()
            num += float(d.norm()) ** 2
            den += float((model.get_submodule(n).weight.detach().float().cpu()
                          - theta0[n].float().cpu()).norm()) ** 2
        frac = changed / max(total, 1)
        surviving = (num ** 0.5) / max(den ** 0.5, 1e-12)
        report["lr_sweep"][f"{lr:g}"] = {
            "changed_fraction": frac,
            "surviving_update_ratio": surviving,
            "displacement_per_weight": lr,
        }
        print(f"[probe] lr={lr:<8g} quantised weights changed: {frac * 100:8.5f}%   "
              f"update surviving quantisation: {surviving:.4f}", flush=True)

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in theta0:
                p.copy_(theta0[n])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[probe] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
