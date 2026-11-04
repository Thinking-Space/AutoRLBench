# Aggregate per-step raw reward (solve rate) from dumped rollout_data .pt files.
# Usage: python3 extract_acc.py <run_dir> <out_csv>
import csv
import glob
import os
import sys

import torch


def iter_samples(x):
    for g in x:
        if isinstance(g, list):
            yield from iter_samples(g)
        else:
            yield g


def main():
    run_dir, out_csv = sys.argv[1:3]
    rows = []
    for path in sorted(
        glob.glob(os.path.join(run_dir, "dump/rollout_data/*.pt")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    ):
        step = int(os.path.basename(path).split(".")[0])
        d = torch.load(path, map_location="cpu", weights_only=False)
        accs = []
        lens = []
        for s in iter_samples(d["samples"]):
            r = s.get("reward")
            if isinstance(r, dict):
                accs.append(1.0 if r.get("acc") else 0.0)
            lens.append(s.get("response_length", 0))
        if accs:
            rows.append(
                {
                    "step": step,
                    "n": len(accs),
                    "solve_rate": round(sum(accs) / len(accs), 4),
                    "mean_response_len": round(sum(lens) / len(lens), 1),
                }
            )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "n", "solve_rate", "mean_response_len"])
        w.writeheader()
        w.writerows(rows)
    sr = [r["solve_rate"] for r in rows]
    k = max(1, len(sr) // 10)
    print(
        f"{run_dir}: steps={len(sr)} first10_mean={sum(sr[:k])/k:.4f} "
        f"last10_mean={sum(sr[-k:])/k:.4f} overall={sum(sr)/len(sr):.4f}"
    )


if __name__ == "__main__":
    main()
