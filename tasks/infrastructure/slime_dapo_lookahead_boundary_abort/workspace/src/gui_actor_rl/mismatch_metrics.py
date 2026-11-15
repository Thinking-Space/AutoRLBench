"""Exact train-vs-rollout log-prob mismatch diagnostics for one rollout batch (monitoring only).

slime's own mismatch metrics (examples/train_infer_mismatch_helper/mis.py + loss.py) are all
means, reduced per token/sequence, so they cannot show a maximum, a tail or a correlation. The
literature says the tail is what matters: the mean |delta| stays small while a few tokens reach
|delta| ~ 1 (arXiv 2605.14220 Fig. 1), and the signed bias, not the magnitude, is what
accumulates (arXiv 2609.20807). This module pools every loss-masked response token of the
rollout (all DP ranks) and logs exact statistics under ``mismatch/*``.

Hooked in through slime's ``--rollout-data-postprocess-path`` (runs after the Megatron
log-prob recompute, before training; it does not touch rollout_data) so no slime core change is
needed:  --rollout-data-postprocess-path gui_actor_rl.mismatch_metrics.log_mismatch_metrics

Convention: delta = log p_train - log p_rollout on the SAMPLED token (same sign as arXiv
2605.14220's delta_t). Positive: the trainer finds the sampled token more likely than the
sampler did. Note mis.py's ``mis_kl`` has the opposite sign (rollout - train).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

ABS_THRESHOLDS = (0.01, 0.1, 0.5, 1.0)
EARLY_POS = 16  # "early" response tokens for the position split
LOW_PROB = 0.1  # rollout probability below which a token counts as low-probability
TOKEN_RATIO_HI = 2.0  # == tis_upper_bound in the run config / tau_tok in arXiv 2605.14220
TOKEN_RATIO_LO = 0.5  # == tis_lower_bound
NIMG_BUCKETS = (("le1", 0, 1), ("2_4", 2, 4), ("5p", 5, 10**9))  # (name, lo, hi) inclusive
_WARNED_CP = False


def _fmt(x: float) -> str:
    return str(x).replace(".", "p")


def compute_mismatch_metrics(
    train_lps: Sequence[np.ndarray],
    rollout_lps: Sequence[np.ndarray],
    loss_masks: Sequence[np.ndarray],
    *,
    advantages: Sequence[np.ndarray] | None = None,
    n_images: Sequence[int] | None = None,
    token_ids: Sequence[np.ndarray] | None = None,
    sample_ids: Sequence[int] | None = None,
    top_k: int = 20,
) -> tuple[dict[str, float], list[dict]]:
    """Pooled statistics over all loss-masked tokens of the given samples.

    Every sequence must be aligned 1:1 (train, rollout, mask [, advantages, token_ids] all of
    the response length); a sample that is not is excluded and counted in ``n_seqs_bad_shape``
    -- a non-zero value there is an indexing bug, not noise.

    Returns (metrics, worst) where ``worst`` lists the ``top_k`` tokens by |delta|.
    """
    n = len(train_lps)
    assert len(rollout_lps) == n and len(loss_masks) == n, (n, len(rollout_lps), len(loss_masks))
    metrics: dict[str, float] = {"n_seqs_total": float(n)}

    bad_shape = nonfinite = positive_lp = zero_rollout_seqs = empty_seqs = 0
    d_parts, lt_parts, lr_parts, sid_parts, pos_parts, adv_parts, nimg_parts, tok_parts = ([] for _ in range(8))
    seq_sum_d, seq_absmax, seq_absmean, seq_k3, seq_len = [], [], [], [], []

    for i in range(n):
        lt, lr, m = (np.asarray(a, dtype=np.float64) for a in (train_lps[i], rollout_lps[i], loss_masks[i]))
        adv = None if advantages is None else np.asarray(advantages[i], dtype=np.float64)
        tok = None if token_ids is None else np.asarray(token_ids[i])
        if not (lt.shape == lr.shape == m.shape) or lt.ndim != 1:
            bad_shape += 1
            continue
        if (adv is not None and adv.shape != lt.shape) or (tok is not None and tok.shape != lt.shape):
            bad_shape += 1
            continue
        valid = m > 0.5
        if not valid.any():
            empty_seqs += 1
            continue
        finite = np.isfinite(lt) & np.isfinite(lr)
        nonfinite += int((valid & ~finite).sum())
        valid &= finite
        if not valid.any():
            empty_seqs += 1
            continue
        pos = np.arange(lt.shape[0])[valid]
        lt_v, lr_v = lt[valid], lr[valid]
        positive_lp += int(((lt_v > 1e-4) | (lr_v > 1e-4)).sum())
        if np.all(lr_v == 0.0):  # placeholder zeros instead of real SGLang log-probs
            zero_rollout_seqs += 1
        d = lt_v - lr_v

        d_parts.append(d)
        lt_parts.append(lt_v)
        lr_parts.append(lr_v)
        sid_parts.append(np.full(d.shape, i))
        pos_parts.append(pos)
        if adv is not None:
            adv_parts.append(adv[valid])
        if tok is not None:
            tok_parts.append(tok[valid])
        nimg_parts.append(np.full(d.shape, -1 if n_images is None else int(n_images[i])))
        seq_sum_d.append(d.sum())
        seq_absmax.append(np.abs(d).max())
        seq_absmean.append(np.abs(d).mean())
        seq_k3.append((np.exp(d) - d - 1.0).sum())
        seq_len.append(d.shape[0])

    metrics.update(
        n_seqs_bad_shape=float(bad_shape),
        n_seqs_empty=float(empty_seqs),
        n_seqs_rollout_lp_all_zero=float(zero_rollout_seqs),
        n_tokens_nonfinite=float(nonfinite),
        n_tokens_logprob_positive=float(positive_lp),
    )
    if not d_parts:
        metrics.update(n_seqs=0.0, n_tokens=0.0)
        return metrics, []

    d = np.concatenate(d_parts)
    lt = np.concatenate(lt_parts)
    lr = np.concatenate(lr_parts)
    sid = np.concatenate(sid_parts)
    pos = np.concatenate(pos_parts)
    ad = np.abs(d)
    pt, pr = np.exp(lt), np.exp(lr)
    ratio = np.exp(d)  # r_corr = p_train / p_rollout
    metrics.update(n_seqs=float(len(seq_len)), n_tokens=float(d.size), resp_len_mean=float(np.mean(seq_len)))

    # --- token level -------------------------------------------------------------------------
    metrics.update(
        delta_mean=float(d.mean()),  # signed: the bias that accumulates
        delta_std=float(d.std()),
        delta_min=float(d.min()),
        delta_max=float(d.max()),
        delta_abs_mean=float(ad.mean()),
        delta_abs_max=float(ad.max()),
    )
    for q in (50, 90, 99, 99.9):
        metrics[f"delta_abs_p{_fmt(q)}"] = float(np.percentile(ad, q))
    for t in ABS_THRESHOLDS:
        metrics[f"frac_abs_gt_{_fmt(t)}"] = float((ad > t).mean())
    # probability space (verl-style rollout_probs_diff_*, rollout_actor_probs_pearson_corr)
    pdiff = np.abs(pt - pr)
    metrics.update(prob_abs_diff_mean=float(pdiff.mean()), prob_abs_diff_max=float(pdiff.max()),
                   prob_abs_diff_std=float(pdiff.std()))
    if pt.std() > 0 and pr.std() > 0:
        metrics["prob_pearson"] = float(np.corrcoef(pt, pr)[0, 1])
    if lt.std() > 0 and lr.std() > 0:
        metrics["logprob_pearson"] = float(np.corrcoef(lt, lr)[0, 1])
    # truncated-IS view: how many tokens a tis_upper/lower_bound of 2.0/0.5 would clip
    metrics.update(frac_ratio_gt_hi=float((ratio > TOKEN_RATIO_HI).mean()),
                   frac_ratio_lt_lo=float((ratio < TOKEN_RATIO_LO).mean()))
    k3 = ratio - d - 1.0
    metrics.update(k1_mean=float((-d).mean()), k3_mean=float(k3.mean()), k3_max=float(k3.max()))

    # --- sequence level (accumulated over the response, what sequence-level IS/RS sees) ------
    s_sum, s_amax, s_amean, s_k3 = (np.asarray(x) for x in (seq_sum_d, seq_absmax, seq_absmean, seq_k3))
    metrics.update(
        seq_delta_sum_mean=float(s_sum.mean()),
        seq_delta_sum_abs_mean=float(np.abs(s_sum).mean()),
        seq_delta_sum_abs_max=float(np.abs(s_sum).max()),
        seq_k1_sum_mean=float((-s_sum).mean()),
        seq_k3_sum_mean=float(s_k3.mean()),
        seq_k3_sum_max=float(s_k3.max()),
        seq_abs_delta_max_mean=float(s_amax.mean()),  # average sequence's worst token
        seq_abs_delta_mean_max=float(s_amean.max()),  # worst sequence's average token
        frac_seqs_with_token_abs_gt_0p5=float((s_amax > 0.5).mean()),
    )

    # --- conditional breakdowns --------------------------------------------------------------
    def split(prefix: str, sel: np.ndarray, signed: bool = False) -> None:
        if sel.any():
            metrics[f"delta_abs_mean_{prefix}"] = float(ad[sel].mean())
            if signed:
                metrics[f"delta_mean_{prefix}"] = float(d[sel].mean())
            metrics[f"n_tokens_{prefix}"] = float(sel.sum())

    split(f"pos_lt{EARLY_POS}", pos < EARLY_POS)
    split(f"pos_ge{EARLY_POS}", pos >= EARLY_POS)
    split("pos_0", pos == 0)
    split(f"rollout_p_lt_{_fmt(LOW_PROB)}", pr < LOW_PROB, signed=True)
    split("rollout_p_ge_0p5", pr >= 0.5, signed=True)
    if n_images is not None:
        ni = np.concatenate(nimg_parts)
        for name, lo, hi in NIMG_BUCKETS:
            split(f"nimg_{name}", (ni >= lo) & (ni <= hi))
    if advantages is not None and adv_parts:
        av = np.concatenate(adv_parts)
        pos_a, neg_a = av > 0, av < 0
        split("adv_pos", pos_a, signed=True)
        split("adv_neg", neg_a, signed=True)
        if pos_a.any() and neg_a.any():
            # arXiv 2609.20807: the trainer-referenced update separates by advantage sign
            metrics["delta_mean_adv_gap"] = float(d[pos_a].mean() - d[neg_a].mean())
        metrics["frac_tokens_adv_zero"] = float((av == 0).mean())

    # --- worst tokens ------------------------------------------------------------------------
    k = min(top_k, d.size)
    idx = np.argpartition(-ad, k - 1)[:k]
    idx = idx[np.argsort(-ad[idx])]
    tok_all = np.concatenate(tok_parts) if tok_parts else None
    adv_all = np.concatenate(adv_parts) if adv_parts else None
    ni_all = np.concatenate(nimg_parts) if n_images is not None else None
    worst = []
    for j in idx:
        row = {
            "delta": float(d[j]),
            "train_p": float(pt[j]),
            "rollout_p": float(pr[j]),
            "seq": int(sid[j]) if sample_ids is None else int(sample_ids[int(sid[j])]),
            "pos": int(pos[j]),
        }
        if tok_all is not None:
            row["token_id"] = int(tok_all[j])
        if adv_all is not None:
            row["advantage"] = float(adv_all[j])
        if ni_all is not None:
            row["n_images"] = int(ni_all[j])
        worst.append(row)
    return metrics, worst


def _n_images(mm) -> int:
    """Number of images in one sample's multimodal_train_inputs (-1 if unknown)."""
    if not mm:
        return 0
    grid = mm.get("image_grid_thw") if hasattr(mm, "get") else None
    if grid is None:
        return -1
    return int(len(grid))


def _to_np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float64)


def _collect_local(rollout_data) -> dict | None:
    """This rank's samples as CPU numpy, or None when the batch has no train/rollout log-probs."""
    if "log_probs" not in rollout_data or "rollout_log_probs" not in rollout_data:
        return None
    n = len(rollout_data["log_probs"])
    resp_lens = rollout_data["response_lengths"]
    tokens = rollout_data.get("tokens")
    payload = {
        "train": [_to_np(t) for t in rollout_data["log_probs"]],
        "rollout": [_to_np(t) for t in rollout_data["rollout_log_probs"]],
        "mask": [_to_np(t) for t in rollout_data["loss_masks"]],
        "adv": [_to_np(t) for t in rollout_data["advantages"]] if "advantages" in rollout_data else None,
        "nimg": None,
        "token_ids": None,
        "sample_ids": None,
    }
    mm = rollout_data.get("multimodal_train_inputs")
    if mm is not None and len(mm) == n:
        payload["nimg"] = [_n_images(x) for x in mm]
    if tokens is not None and len(tokens) == n:
        # the log-prob at response position j belongs to the token tokens[-response_length + j]
        payload["token_ids"] = [
            tokens[i][-int(resp_lens[i]) :].detach().cpu().numpy() if resp_lens[i] > 0 else np.zeros(0, dtype=np.int64)
            for i in range(n)
        ]
    sidx = rollout_data.get("sample_indices")
    if sidx is not None and len(sidx) == n:
        payload["sample_ids"] = [int(x) for x in sidx]
    return payload


def _merge(payloads: list[dict]) -> dict:
    out = {}
    for key in ("train", "rollout", "mask", "adv", "nimg", "token_ids", "sample_ids"):
        parts = [p[key] for p in payloads]
        # a field is usable only if every rank has it
        out[key] = None if any(x is None for x in parts) else [item for x in parts for item in x]
    return out


def log_mismatch_metrics(args, rollout_id: int, rollout_data) -> None:
    """slime ``--rollout-data-postprocess-path`` hook. Never modifies ``rollout_data``."""
    global _WARNED_CP
    if not getattr(args, "get_mismatch_metrics", True):
        return
    import torch.distributed as dist
    from megatron.core import mpu

    # Same ranks as slime's log_rollout_data: TP rank 0 of the last PP stage hold log-probs.
    if not (mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()):
        return
    if mpu.get_context_parallel_world_size() > 1:
        if not _WARNED_CP:
            logger.warning("mismatch_metrics: context parallel > 1 is not supported, skipping")
            _WARNED_CP = True
        return

    payload = None
    try:
        payload = _collect_local(rollout_data)
    except Exception:  # monitoring must never take the run down
        logger.exception("mismatch_metrics: collecting local data failed")

    # every rank of the DP group takes part in the gather, whether or not it has data
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    dp_src = mpu.get_data_parallel_src_rank(with_context_parallel=True)
    dp_group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
    if dist.get_rank() != dp_src:
        dist.gather_object(payload, None, dst=dp_src, group=dp_group)
        return
    gathered = [None] * dp_size
    dist.gather_object(payload, gathered, dst=dp_src, group=dp_group)

    try:
        have = [p for p in gathered if p is not None]
        if len(have) != dp_size:
            logger.warning("mismatch_metrics: %d/%d DP ranks lack log_probs/rollout_log_probs", dp_size - len(have), dp_size)
            if not have:
                return
        m = _merge(have)
        metrics, worst = compute_mismatch_metrics(
            m["train"], m["rollout"], m["mask"], advantages=m["adv"], n_images=m["nimg"],
            token_ids=m["token_ids"], sample_ids=m["sample_ids"],
        )
        _emit(args, rollout_id, metrics, worst)
    except Exception:
        logger.exception("mismatch_metrics: computing/logging failed")


def _emit(args, rollout_id: int, metrics: dict, worst: list[dict]) -> None:
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    logger.info("mismatch %d: %s", rollout_id, {k: round(v, 6) for k, v in metrics.items()})
    if metrics.get("n_seqs_bad_shape", 0) > 0 or metrics.get("n_tokens_nonfinite", 0) > 0:
        logger.error("mismatch %d: INDEXING/NUMERIC PROBLEM n_seqs_bad_shape=%s n_tokens_nonfinite=%s", rollout_id,
                     metrics["n_seqs_bad_shape"], metrics["n_tokens_nonfinite"])

    log_dict = {f"mismatch/{k}": v for k, v in metrics.items() if np.isfinite(v)}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, log_dict, step_key="rollout/step")

    dump_dir = getattr(args, "gui_mismatch_dump_dir", None)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        record = {"rollout_id": rollout_id, "metrics": metrics, "worst_tokens": worst}
        with open(os.path.join(dump_dir, f"rollout_{rollout_id:06d}.json"), "w") as f:
            json.dump(record, f, indent=1)
        with open(os.path.join(dump_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"rollout_id": rollout_id, **metrics}) + "\n")
