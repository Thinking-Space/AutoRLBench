"""On the rounding-informative elements S (bf16(W_base + dW) != bf16(W_base)), measure how far the received bf16
weight is from bf16(W_base + dW) in bf16 ulps (distance between the int16 bit patterns of same-sign values).
Usage: python probe_ulp.py <probe dir>"""
import collections
import glob
import os
import sys

import torch


def ulps(a, b):
    ia, ib = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    same_sign = (ia >= 0) == (ib >= 0)
    return torch.where(same_sign, (ia - ib).abs(), torch.full_like(ia, 10**6))


def main():
    d = sys.argv[1]
    actor = {}
    for p in glob.glob(os.path.join(d, "actor_k*.pt")):
        r = torch.load(p)
        actor[(r["k"], r["name"])] = r
    total = collections.Counter()
    for p in sorted(glob.glob(os.path.join(d, "recv_k*.pt"))):
        r = torch.load(p)
        a = actor.get((r["k"], r["name"]))
        if a is None or r["k"] == 0:
            continue
        recv = r["recv"].bfloat16()
        base = a["w_base"].bfloat16()
        merged = (a["w_base"] + a["delta"]).bfloat16()
        s = merged != base
        dist = ulps(recv[s], merged[s])
        c = collections.Counter(min(int(x), 3) for x in dist.tolist())
        total.update(c)
        n = int(s.sum())
        print(f"k={r['k']} {r['name']:<32} |S|={n:8d} ulp distance to bf16(Wb+dW): "
              f"0:{c[0] / n:.4f} 1:{c[1] / n:.4f} 2:{c[2] / n:.4f} >=3:{c[3] / n:.4f}")
    n = sum(total.values())
    print(f"ALL |S|={n}: " + " ".join(f"{k if k < 3 else '>=3'}:{total[k] / n:.5f}" for k in range(4)))


if __name__ == "__main__":
    main()
