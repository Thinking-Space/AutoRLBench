"""Measure how far the training sequence drifts from the rollout token stream.

Reads the episode dumps written by calc_agent (CALC_AGENT_DUMP_DIR/episodes.*.jsonl), where each
record holds the token stream exchanged with vLLM (stream_*) and the sequence the agent loop
returned to the trainer (prompt_ids / response_*).

Token-level (CPU):
  * identical: trained response == stream response, token for token (mask included)
  * dropped:   fraction of sampled tokens (stream mask=1) that do not appear in the trained sequence
               (difflib alignment of the two response sequences)
  * novel:     fraction of trained tokens (mask=1) that were never sampled at that place
  * positional mismatch: fraction of mask=1 positions k where the k-th trained token differs from
               the k-th sampled token (this is the pairing the rollout log-probs are carried over by)

Log-prob level (GPU, --model): recompute log-probs with the HF model (bf16) on
  (a) prompt + stream response    -> compare with the vLLM log-probs (numerics floor)
  (b) prompt + trained response   -> compare with the log-probs handed to the trainer
  (c) aligned sampled tokens only -> same token, rollout context vs trained context
and report mean |diff|, KL(rollout||train) = E[lp_rollout - lp_train], k3, and the token ratio
exp(lp_train - lp_rollout) quantiles.  Use dumps from the first training step so the weights
the logprobs were sampled with equal the base checkpoint.
"""

import argparse
import difflib
import glob
import json
import math
import os
import statistics


def load(dump_dir, split, steps):
    recs = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "episodes.*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if split and r.get("split") != split:
                    continue
                if steps is not None and r.get("global_steps") not in steps:
                    continue
                recs.append(r)
    return recs


def token_metrics(r):
    s_ids, s_mask = r["stream_response_ids"], r["stream_response_mask"]
    t_ids, t_mask = r["response_ids"], r["response_mask"]
    s_gen = [t for t, m in zip(s_ids, s_mask) if m]
    t_gen = [t for t, m in zip(t_ids, t_mask) if m]
    sm = difflib.SequenceMatcher(a=s_ids, b=t_ids, autojunk=False)
    pairs = []
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            if s_mask[a + k] and t_mask[b + k]:
                pairs.append((a + k, b + k))
    k_min = min(len(s_gen), len(t_gen))
    pos_mis = sum(1 for k in range(k_min) if s_gen[k] != t_gen[k]) + abs(len(s_gen) - len(t_gen))
    return {
        "identical": s_ids == t_ids and list(s_mask) == list(t_mask),
        "n_sampled": len(s_gen),
        "n_trained": len(t_gen),
        "dropped": 1 - len(pairs) / max(1, len(s_gen)),
        "novel": 1 - len(pairs) / max(1, len(t_gen)),
        "positional_mismatch": pos_mis / max(1, max(len(s_gen), len(t_gen))),
        "pairs": pairs,
    }


def summarize(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals_sorted = sorted(vals)

    def q(p):
        return vals_sorted[min(len(vals_sorted) - 1, int(p * (len(vals_sorted) - 1)))]

    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "p05": q(0.05),
        "p50": q(0.5),
        "p95": q(0.95),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


def logprob_metrics(recs, model_path, device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def lp(prompt, resp):
        ids = torch.tensor([prompt + resp], device=device)
        logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
        return torch.log_softmax(logits, -1).gather(-1, ids[0, len(prompt) :, None])[:, 0].cpu().tolist()

    out = {"stream": [], "trained": [], "aligned": [], "seq_logratio_trained": [], "seq_logratio_aligned": []}
    for r in recs:
        tm = r["_tok"]
        s_lp = lp(r["prompt_ids"], r["stream_response_ids"])
        t_lp = lp(r["prompt_ids"], r["response_ids"])
        for k, m in enumerate(r["stream_response_mask"]):
            if m:
                out["stream"].append((r["stream_logprobs"][k], s_lp[k]))
        seq = 0.0
        for k, m in enumerate(r["response_mask"]):
            if m and r["response_logprobs"] is not None:
                out["trained"].append((r["response_logprobs"][k], t_lp[k]))
                seq += t_lp[k] - r["response_logprobs"][k]
        out["seq_logratio_trained"].append(seq)
        seq = 0.0
        for i, j in tm["pairs"]:
            out["aligned"].append((r["stream_logprobs"][i], t_lp[j]))
            seq += t_lp[j] - r["stream_logprobs"][i]
        out["seq_logratio_aligned"].append(seq)

    def stats(pairs):
        if not pairs:
            return None
        d = [t - s for s, t in pairs]  # log ratio train/rollout
        return {
            "tokens": len(pairs),
            "mean_abs_logprob_diff": statistics.fmean(abs(x) for x in d),
            "kl_rollout_train": statistics.fmean(-x for x in d),
            "k3": statistics.fmean(math.exp(x) - x - 1 for x in d),
            "ratio": summarize([math.exp(max(-20, min(20, x))) for x in d]),
            "frac_ratio_outside_0.8_1.2": statistics.fmean(1.0 if not (0.8 <= math.exp(x) <= 1.2) else 0.0 for x in d),
        }

    return {
        "stream_vs_vllm": stats(out["stream"]),
        "trained_vs_logprobs_given_to_trainer": stats(out["trained"]),
        "aligned_same_token_rollout_ctx_vs_trained_ctx": stats(out["aligned"]),
        "seq_logratio_trained": summarize(out["seq_logratio_trained"]),
        "seq_logratio_aligned": summarize(out["seq_logratio_aligned"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", default=None, help="comma list of global_steps to keep")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-episodes", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    steps = None if args.steps is None else {int(s) for s in args.steps.split(",")}
    recs = load(args.dump_dir, args.split, steps)
    for r in recs:
        r["_tok"] = token_metrics(r)
    result = {
        "dump_dir": args.dump_dir,
        "split": args.split,
        "steps": sorted(steps) if steps else None,
        "episodes": len(recs),
        "identical_frac": statistics.fmean(1.0 if r["_tok"]["identical"] else 0.0 for r in recs) if recs else None,
        "sampled_tokens_per_episode": summarize([r["_tok"]["n_sampled"] for r in recs]),
        "trained_tokens_per_episode": summarize([r["_tok"]["n_trained"] for r in recs]),
        "dropped_sampled_token_frac": summarize([r["_tok"]["dropped"] for r in recs]),
        "novel_trained_token_frac": summarize([r["_tok"]["novel"] for r in recs]),
        "positional_mismatch_frac": summarize([r["_tok"]["positional_mismatch"] for r in recs]),
        "tokens_total": {
            "sampled": sum(r["_tok"]["n_sampled"] for r in recs),
            "trained": sum(r["_tok"]["n_trained"] for r in recs),
            "sampled_and_trained": sum(len(r["_tok"]["pairs"]) for r in recs),
        },
    }
    if args.model:
        result["logprobs"] = logprob_metrics(recs[: args.max_episodes], args.model, args.device)
        result["logprobs"]["episodes"] = min(len(recs), args.max_episodes)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
