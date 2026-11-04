"""Compare what vLLM received with what the actor should have sent, per sync and layer.
  dW   = scaling * B @ A of the live adapter at sync k (actor side, fp32)
  recv = tensor vLLM's load_weights received at sync k (consumer side, bf16)
The merge happens in bf16, and most per-element dW are below half a bf16 ulp of W, so a correctly merged weight
differs from bf16(W_base) only where rounding flips. The informative elements are therefore
  S = { i : bf16(W_base + dW)_i != bf16(W_base)_i }
and on S we count whether recv equals the merged value or the base value.
Columns: |dW| (fp32 Frobenius), |S|, share of S where recv == bf16(W_base + dW), share where recv == bf16(W_base),
         and whether recv is bit-identical to bf16(W_base) over the whole tensor.
Usage: python probe_report.py <probe dir> [<probe dir> ...]"""
import glob
import os
import re
import sys

import torch


def main():
    for d in sys.argv[1:]:
        print(f"=== {d}")
        actor = {}
        for p in glob.glob(os.path.join(d, "actor_k*.pt")):
            r = torch.load(p)
            actor[(r["k"], r["name"])] = r
        print(f"{'k':>3} {'layer':<32} {'|dW|':>8} {'|S|':>8} {'recv=merged':>12} {'recv=base':>10} {'all==base':>10}  replica")
        for p in sorted(glob.glob(os.path.join(d, "recv_k*.pt"))):
            r = torch.load(p)
            pid = re.search(r"_pid(\d+)_", p).group(1)
            a = actor.get((r["k"], r["name"]))
            if a is None:
                print(f"{r['k']:>3} {r['name']:<32} (no actor record for this sync)")
                continue
            recv = r["recv"].bfloat16()
            base = a["w_base"].bfloat16()
            merged = (a["w_base"] + a["delta"]).bfloat16()
            s = merged != base
            n_s = int(s.sum())
            to_merged = (recv[s] == merged[s]).float().mean().item() if n_s else float("nan")
            to_base = (recv[s] == base[s]).float().mean().item() if n_s else float("nan")
            print(f"{r['k']:>3} {r['name']:<32} {a['delta'].norm().item():8.4f} {n_s:8d} {to_merged:12.4f} {to_base:10.4f} "
                  f"{str(torch.equal(recv, base)):>10}  {pid}")


if __name__ == "__main__":
    main()
