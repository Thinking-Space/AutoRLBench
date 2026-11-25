#!/usr/bin/env python3
"""Sliding-window perplexity of a scoring model over stored trajectories.

One teacher-forced forward per trajectory gives the per-token NLL under full
left context; the stride-1 sliding window is then a cumulative sum over that
array rather than 31k separate forwards.  This is *not* the same quantity as
re-conditioning each 1024-token window on itself -- every window here keeps the
full prefix -- which is what we want: the question is how a token's cost under
the scoring model drifts as the generated prefix grows, and truncating the
prefix would throw exactly that away.

The scoring model can be run with a fake-quantised KV cache so that "the
student" means the student as deployed, not an fp16 shadow of it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


def _load_trajectories(path: str, lo: int, hi: int, limit: int):
    import pyarrow.parquet as pq

    cols = pq.read_table(path).to_pydict()
    rows = []
    for i in range(len(cols["example_id"])):
        n = int(cols["response_tokens"][i])
        if n < lo or n > hi:
            continue
        rows.append({
            "example_id": cols["example_id"][i],
            "prompt_token_ids": list(cols["prompt_token_ids"][i]),
            "response_token_ids": list(cols["response_token_ids"][i]),
            "response_tokens": n,
            "finish_reason": cols["finish_reason"][i],
        })
    rows.sort(key=lambda r: r["example_id"])
    if limit > 0:
        rows = rows[:limit]
    return rows


def _install_kv_quant(model, scheme: str, attn: str) -> None:
    if scheme in ("", "none", "auto", "bf16"):
        return
    if scheme.startswith("turboquant"):
        # The installer lives in turboquant_attn_hook; turboquant_fake_quant
        # only holds the tensor-level kernels it calls.
        from opqad.integrations.turboquant_attn_hook import install_turboquant_fake_quant
        install_turboquant_fake_quant(model, scheme, base_impl=attn)
    elif scheme.startswith("kivi"):
        from opqad.quantization.kivi_fake_quant import install_kivi_fake_quant, KiviFakeQuantConfig
        bits = int(scheme.replace("kivi_", "").replace("bit", "") or 2)
        install_kivi_fake_quant(
            model,
            KiviFakeQuantConfig(bits=bits, group_size=32, residual_length=32),
            base_impl=attn,
        )
    else:
        raise SystemExit(f"unknown --kv-scheme {scheme!r}")


@torch.no_grad()
def _token_nll(model, ids: list[int], n_prompt: int, chunk: int) -> np.ndarray:
    """NLL of every response token under full left context, in nats."""
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state[0]

    # position t of `hidden` predicts token t+1, so the response tokens are
    # predicted by hidden[n_prompt-1 : len(ids)-1]
    start = n_prompt - 1
    stop = len(ids) - 1
    out = torch.empty(stop - start, dtype=torch.float32, device=device)
    for a in range(start, stop, chunk):
        b = min(a + chunk, stop)
        logits = model.lm_head(hidden[a:b]).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = x[0, a + 1 : b + 1]
        out[a - start : b - start] = -logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        del logits, logprobs
    return out.cpu().numpy().astype(np.float32)


def _sliding(nll: np.ndarray, window: int) -> np.ndarray:
    """Stride-1 mean NLL over `window` tokens; empty if the run is too short."""
    if nll.size < window:
        return np.empty(0, dtype=np.float32)
    c = np.concatenate([[0.0], np.cumsum(nll, dtype=np.float64)])
    return ((c[window:] - c[:-window]) / window).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the scoring model")
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True, help="output .npz; a .json summary is written alongside")
    ap.add_argument("--kv-scheme", default="none",
                    help="fake-quant KV cache for the scoring model; 'none' scores in bf16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--min-response-tokens", type=int, default=2048,
                    help="the 8k analysis band is applied downstream, so score wider here")
    ap.add_argument("--max-response-tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--logits-chunk", type=int, default=2048,
                    help="tokens per lm_head call; the full 32k x 152k logit tensor does not fit")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = _load_trajectories(
        args.trajectories, args.min_response_tokens, args.max_response_tokens, args.limit)
    if not rows:
        raise SystemExit(
            f"no trajectory in [{args.min_response_tokens}, {args.max_response_tokens}] tokens")
    print(f"[ppl] {len(rows)} trajectories, "
          f"lengths {min(r['response_tokens'] for r in rows)}-"
          f"{max(r['response_tokens'] for r in rows)}", flush=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn,
        trust_remote_code=True).cuda().eval()
    _install_kv_quant(model, args.kv_scheme, args.attn)

    per_traj, curves = [], {}
    started = time.time()
    for i, r in enumerate(rows):
        ids = r["prompt_token_ids"] + r["response_token_ids"]
        nll = _token_nll(model, ids, len(r["prompt_token_ids"]), args.logits_chunk)
        curve = _sliding(nll, args.window)
        curves[f"nll_{r['example_id']}"] = nll.astype(np.float16)
        per_traj.append({
            "example_id": r["example_id"],
            "response_tokens": r["response_tokens"],
            "finish_reason": r["finish_reason"],
            "ppl_full": float(math.exp(float(nll.mean()))),
            "ppl_first_window": float(math.exp(float(curve[0]))) if curve.size else None,
            "ppl_last_window": float(math.exp(float(curve[-1]))) if curve.size else None,
        })
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"[ppl] {i + 1}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    # Two aggregate curves.  Absolute position answers "does cost grow with
    # context length"; decile position answers it without letting the longest
    # trajectories be the only ones present in the tail of the mean.
    max_len = max(max(len(curves[f"nll_{r['example_id']}"]) - args.window + 1 for r in rows), 0)
    total = np.zeros(max_len, dtype=np.float64)
    count = np.zeros(max_len, dtype=np.int64)
    deciles = np.zeros((len(rows), 10), dtype=np.float64)
    for j, r in enumerate(rows):
        nll = np.asarray(curves[f"nll_{r['example_id']}"], dtype=np.float32)
        curve = _sliding(nll, args.window)
        total[: curve.size] += curve
        count[: curve.size] += 1
        if curve.size:
            edges = np.linspace(0, curve.size, 11).astype(int)
            for d in range(10):
                a, b = edges[d], max(edges[d + 1], edges[d] + 1)
                deciles[j, d] = curve[a:b].mean()

    mean_curve = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    summary = {
        "label": args.label or Path(args.out).stem,
        "scoring_model": args.model,
        "kv_scheme": args.kv_scheme,
        "trajectories_file": args.trajectories,
        "window": args.window,
        "n_trajectories": len(rows),
        "length_range": [min(r["response_tokens"] for r in rows),
                         max(r["response_tokens"] for r in rows)],
        "mean_response_tokens": float(np.mean([r["response_tokens"] for r in rows])),
        "ppl_full_mean": float(np.mean([p["ppl_full"] for p in per_traj])),
        "ppl_full_median": float(np.median([p["ppl_full"] for p in per_traj])),
        "decile_ppl": [float(math.exp(v)) for v in deciles.mean(axis=0)],
        "decile_ppl_per_traj_ratio_last_over_first": float(
            np.mean(np.exp(deciles[:, -1] - deciles[:, 0]))),
        "elapsed_seconds": time.time() - started,
        "per_trajectory": per_traj,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mean_window_nll=mean_curve.astype(np.float32),
        window_count=count,
        decile_nll=deciles.astype(np.float32),
        example_ids=np.array([r["example_id"] for r in rows]),
        response_tokens=np.array([r["response_tokens"] for r in rows]),
        **curves,
    )
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[ppl] ppl_full mean {summary['ppl_full_mean']:.4f} "
          f"decile1 {summary['decile_ppl'][0]:.4f} -> decile10 {summary['decile_ppl'][-1]:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
