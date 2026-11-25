"""Forward KL amplified where the student has lost the teacher's token.

The objective
-------------
    L = D_KL(p_s || p_t)  +  r * D_KL(p_t || p_s)

    y* = argmax p_t                      the teacher's own best token
    r  = clip(p_t(y*) / p_s(y*), 0, R)   how many times the student underweights it

The reverse term keeps weight 1.  That is the difference from a normalised
mixture `a*rev + (1-a)*fwd`, where a -> 1 deletes the reverse term and the
calibration anchor goes with it; that form oscillated 0.0227% -> 95% -> 12.8%
-> 0.36% -> 68.7% over four steps on the k2v3 probe.  Here the anchor never
leaves and the forward term is added on top.

Why a RATIO and not a difference
--------------------------------
The same objective with `alpha_F = (1 + p_t(y*) - p_s(y*))/2`, i.e. an additive
gap, was measured on the OSCAR-1.5B failure over 585613 positions
(experiments/2026-09-20/token_forensics/REPORT_ALPHA_ZH.md).  It does not
concentrate: terminator-wanted positions are 4.77% of positions and take 5.32%
of the amplification, a factor of 1.11.

The reason is structural.  `g = p_t - p_s <= p_t`, so the difference form obeys

    r_diff = (1+g)/(1-g) <= (1 + p_t(y*)) / (1 - p_t(y*))

-- a ceiling set by the TEACHER alone, which no amount of student error raises.
At the positions where OSCAR loses terminators the teacher averages 0.299, so
r_diff there cannot exceed 1.85 however badly the student fails.  And it does
fail there: student/teacher 0.202 against 0.506 elsewhere, a 2.5x concentration
in the damage that the difference reports as 1.19x.

The damage is multiplicative and the difference measures an additive gap:
0.30 -> 0.06 is 5x as a ratio and 0.24 as a difference.  A ratio has no
teacher-set ceiling, so it sees it.

One clamp bounds both, and it is applied to the FORWARD term only
-----------------------------------------------------------------
`p_s` has no lower bound -- in float32 it underflows to 0 outright -- so
`log p_s = -inf` makes the forward KL `+inf`, and `r = p_t/p_s` blows up on the
SAME token.  Their product goes as `(1/p_s) log(1/p_s)`: capping `r` alone does
not bound it.

Per token,

    d_v       = log p_t(v) - log p_s(v)
    d_fwd,v   = min(d_v, +log R)
    fwd       = sum_v p_t(v) * d_fwd,v
    rev       = sum_v p_s(v) * (-d_v)          <-- UNCLAMPED, on purpose

The positive contribution is bounded by `log R`; the lower side is left open
so `r = min(exp(d at y*), R)` has no artificial lower bound.  This is the
requested ratio semantics.  `rev` remains the exact reverse-KL anchor.

`rev` deliberately does not share the clamp.  It is the anchor of the
objective, and clamping it silently replaces the reverse KL with something that
is not a KL: the term it truncates is exactly the mass the student has put
where the teacher has none, which is the thing the anchor exists to punish.  It
needs no bound of its own -- `log p_t` is finite for every token a log-softmax
produces, and where `p_s` underflows to 0 the contribution is `0 * finite = 0`.
The earlier version of this file clamped both and was wrong.

Flooring `p_s` at eps instead would bound the top at `log(1/eps)` but breaks
normalisation, and the residual mass `V*eps` can drive the sum NEGATIVE -- at
eps=1e-4 over 151936 tokens that is -140 nats.  Clamping the log-ratio has no
such term.

The clamp also does something the cap does not: beyond it the gradient is
EXACTLY zero, so a token the student has no chance on stops being pushed rather
than taking the whole batch.

`r` is detached.  It is a weight chosen from the current state, not a quantity
to optimise; leaving it in the graph would let the model lower the loss by
making its own weight smaller.
"""

import math
from typing import Dict, Tuple

import torch

__all__ = ["ratio_forward_kl"]


def ratio_forward_kl(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    response_mask: torch.Tensor,
    r_max: float = 100.0,
    coef: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``D_KL(p_s||p_t) + r D_KL(p_t||p_s)`` with ``r`` a clipped probability ratio.

    Args:
        student_logp: ``[B, T, V]`` differentiable student log-probabilities,
            normalised over V.  Absolute, not renormalised on a subset: both KLs
            here are expectations under a full distribution.
        teacher_logp: ``[B, T, V]`` frozen teacher log-probabilities, same shape.
        response_mask: ``[B, T]``.
        r_max: the clamp, applied to the ratio and, as ``log r_max``, to the
            per-token log-ratio inside both KLs.  100 corresponds to
            ``p_t/p_s = 100``, i.e. "teacher 0.30, student 0.003" saturates.
        coef: linear scale, already applied to the returned loss.

    Returns:
        ``(loss, metrics)``, token-mean aggregated.  ``metrics`` reports the two
        KLs separately, the mean and saturation rate of ``r``, and how often each
        clamp binds -- a run where the clamps bind often is training on the
        clamp rather than on the objective, and that has to be visible.
    """
    if student_logp.shape != teacher_logp.shape:
        raise ValueError(
            f"student {tuple(student_logp.shape)} and teacher "
            f"{tuple(teacher_logp.shape)} log-probabilities must have the same shape"
        )
    if student_logp.dim() != 3:
        raise ValueError(
            f"expected [B, T, V] log-probabilities, got {tuple(student_logp.shape)}"
        )
    if response_mask.shape != student_logp.shape[:2]:
        raise ValueError(
            f"response_mask must be {student_logp.shape[:2]}, got {tuple(response_mask.shape)}"
        )
    if not r_max > 1.0:
        raise ValueError(f"r_max must exceed 1, got {r_max}")

    ls = student_logp.float()
    # .detach() and not just a no_grad block: `.float()` on a tensor that is
    # already float32 returns the SAME tensor, so `with torch.no_grad(): lt =
    # teacher_logp.float()` leaves `lt is teacher_logp` with its original
    # requires_grad, and the subtraction below then differentiates through the
    # teacher.  test_teacher_is_not_differentiated caught exactly this.
    lt = teacher_logp.detach().float()

    log_r_max = float(torch.log(torch.tensor(r_max)))
    # d_v = log p_t - log p_s.  Non-finite entries mean an underflowed
    # probability on one side; torch.clamp already maps +-inf to the bounds, so
    # no nan_to_num is needed for them -- but -inf minus -inf is nan, so that
    # case is handled.
    d = lt - ls
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    # The clamp applies to the FORWARD term and to r, not to the reverse KL.
    d_forward = d.clamp(max=log_r_max)

    pt = lt.exp()
    ps = ls.exp()
    # forward = E_{p_t}[d], in [-log_r_max, +log_r_max] since sum pt = 1.
    forward = (pt * d_forward).sum(-1)
    # reverse = E_{p_s}[-d], with the UNCLAMPED d.  This term is the anchor and
    # must stay a KL: clamping it truncates exactly the mass the student has
    # put where the teacher has none.  It needs no bound -- log p_t is finite
    # for anything a log-softmax produces, and where p_s underflows to 0 the
    # contribution is 0 * finite = 0.
    reverse = (ps * (-d)).sum(-1)

    with torch.no_grad():
        y_star = pt.argmax(-1, keepdim=True)
    # No lower bound: if the student is sharper than the teacher on y*, the
    # forward term is down-weighted.  r_max is the only cap.
    r = d_forward.gather(-1, y_star).squeeze(-1).exp().clamp(max=r_max).detach()

    per_token = reverse + r * forward
    mask = response_mask.to(per_token.dtype)
    denom = mask.sum()
    if float(denom) == 0.0:
        zero = student_logp.sum() * 0.0
        return zero, {
            k: 0.0
            for k in (
                "ratio_fwd_kl", "ratio_fwd_tokens", "ratio_fwd_forward",
                "ratio_fwd_reverse", "ratio_fwd_r_mean", "ratio_fwd_r_saturated",
                "ratio_fwd_clamp_frac", "ratio_fwd_p_student_at_ystar",
                "ratio_fwd_p_teacher_at_ystar",
            )
        }

    loss = (per_token * mask).sum() / denom
    with torch.no_grad():
        m3 = mask.unsqueeze(-1)
        metrics = {
            "ratio_fwd_kl": float(loss.detach()),
            "ratio_fwd_tokens": float(denom),
            "ratio_fwd_forward": float((forward * mask).sum() / denom),
            "ratio_fwd_reverse": float((reverse * mask).sum() / denom),
            "ratio_fwd_r_mean": float((r * mask).sum() / denom),
            "ratio_fwd_r_saturated": float(
                ((r >= r_max).float() * mask).sum() / denom
            ),
            # How often the per-token clamp binds anywhere in the row.  If this
            # is large the objective is mostly the clamp.
            "ratio_fwd_clamp_frac": float(
                ((d.abs() >= log_r_max).float() * m3).sum() / (denom * d.shape[-1])
            ),
            "ratio_fwd_p_student_at_ystar": float(
                (ps.gather(-1, y_star).squeeze(-1) * mask).sum() / denom
            ),
            "ratio_fwd_p_teacher_at_ystar": float(
                (pt.gather(-1, y_star).squeeze(-1) * mask).sum() / denom
            ),
        }
    return loss * coef, metrics


def load_lm_head(
    model_path: str,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Read the frozen teacher's ``lm_head.weight`` [V, H] without the model.

    Duplicated from ``opqad.losses.fullvocab_kl`` on purpose: this module is the OSCAR
    path and must not break when that one is being rewritten.  Thirty lines is a
    cheaper dependency than a shared file under concurrent edit.

    Raises rather than falling back when the checkpoint ties its embeddings: a
    tied model has no ``lm_head.weight`` and the right tensor is
    ``model.embed_tokens.weight``.  Picking one silently would build a
    plausible-looking teacher out of the wrong matrix.
    """
    import glob
    import json
    import os

    from safetensors.torch import safe_open

    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {model_path}")
    cfg_path = os.path.join(model_path, "config.json")
    tied = False
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            tied = bool(json.load(fh).get("tie_word_embeddings", False))
    want = "model.embed_tokens.weight" if tied else "lm_head.weight"
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as fh:
            if want in fh.keys():
                return fh.get_tensor(want).to(device=device, dtype=dtype).contiguous()
    raise KeyError(
        f"{want!r} not found in any shard under {model_path} "
        f"(tie_word_embeddings={tied}); refusing to guess which matrix is the head"
    )


def _chunk_term(student_logits, teacher_hidden, lm_head, log_r_max, r_max):
    """One chunk of ``reverse + r * forward``.  Returns [b, c] fp32.

    Everything here is transient: the caller wraps it in ``checkpoint`` so the
    two [b, c, V] log-softmaxes are dropped after the forward and rebuilt in
    backward.  A full [b, T, V] teacher logit tensor is never materialised --
    the teacher head is applied inside the chunk.
    """
    logp_s = torch.log_softmax(student_logits.float(), dim=-1)
    t = torch.nn.functional.linear(teacher_hidden, lm_head).float()
    logp_t = torch.log_softmax(t, dim=-1)
    del t

    d = logp_t - logp_s
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    # Clamped for the forward term and r; the reverse KL uses the raw d.
    d_forward = d.clamp(max=log_r_max)
    y_star = logp_t.argmax(dim=-1, keepdim=True)
    r = d_forward.gather(-1, y_star).squeeze(-1).exp().clamp(max=r_max).detach()
    forward = (logp_t.exp() * d_forward).sum(dim=-1)
    reverse = (logp_s.exp() * (-d)).sum(dim=-1)
    return torch.stack([reverse + r * forward, reverse, forward, r], dim=0)


def ratio_forward_kl_from_hidden(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: torch.Tensor,
    response_mask: torch.Tensor,
    r_max: float = 100.0,
    chunk: int = 256,
    coef: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """:func:`ratio_forward_kl` over a full vocabulary, chunked and checkpointed.

    Takes the teacher as ``(post-norm hidden state, lm_head)`` rather than as
    log-probabilities: shipping [B, T, 151936] teacher logits is 18.5 GB at the
    16k context, the hidden state is 0.19 GB, and the head is a single resident
    [V, H] matrix.

    Args:
        student_logits: ``[b, T, V]``, differentiable.
        teacher_hidden: ``[b, T, H]``, the FINAL POST-NORM hidden state -- the
            tensor ``lm_head`` consumes.  A pre-norm hidden state produces a
            well-formed number that is not the teacher.
        lm_head: ``[V, H]``, frozen.
        response_mask: ``[b, T]``.
        r_max: the clamp.  Bounds ``forward`` to ``[-log r_max, log r_max]`` and
            ``r`` to ``r_max``; see the module docstring for why one clamp has
            to cover both.
        chunk: positions per checkpointed chunk.  Transient is about
            ``b * chunk * V * 10`` bytes.
        coef: linear scale, already applied to the returned loss.
    """
    if student_logits.dim() != 3:
        raise ValueError(f"student_logits must be [b, T, V]; got {tuple(student_logits.shape)}")
    if teacher_hidden.dim() != 3:
        raise ValueError(f"teacher_hidden must be [b, T, H]; got {tuple(teacher_hidden.shape)}")
    if lm_head.dim() != 2:
        raise ValueError(f"lm_head must be [V, H]; got {tuple(lm_head.shape)}")
    b, t_len, vocab = student_logits.shape
    if teacher_hidden.shape[:2] != (b, t_len):
        raise ValueError(
            f"teacher_hidden {tuple(teacher_hidden.shape)} does not align with "
            f"student_logits {tuple(student_logits.shape)} on [b, T]"
        )
    if lm_head.shape[0] != vocab:
        raise ValueError(f"lm_head vocab {lm_head.shape[0]} != student vocab {vocab}")
    if lm_head.shape[1] != teacher_hidden.shape[2]:
        raise ValueError(
            f"lm_head hidden {lm_head.shape[1]} != teacher_hidden hidden {teacher_hidden.shape[2]}"
        )
    if response_mask.shape != (b, t_len):
        raise ValueError(f"response_mask must be {(b, t_len)}; got {tuple(response_mask.shape)}")
    if not r_max > 1.0:
        raise ValueError(f"r_max must exceed 1, got {r_max}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive; got {chunk}")

    from torch.utils.checkpoint import checkpoint

    mask = response_mask.to(torch.float32)
    denom = mask.sum()
    if float(denom) == 0.0:
        zero = student_logits.sum() * 0.0
        return zero, {
            k: 0.0
            for k in (
                "ratio_fwd_kl", "ratio_fwd_tokens", "ratio_fwd_forward",
                "ratio_fwd_reverse", "ratio_fwd_r_mean", "ratio_fwd_r_saturated",
            )
        }

    log_r_max = math.log(r_max)
    grad_on = torch.is_grad_enabled() and student_logits.requires_grad
    total = student_logits.new_zeros((), dtype=torch.float32)
    sums = {"reverse": 0.0, "forward": 0.0, "r": 0.0, "r_sat": 0.0}
    for lo in range(0, t_len, chunk):
        hi = min(lo + chunk, t_len)
        args = (student_logits[:, lo:hi], teacher_hidden[:, lo:hi], lm_head,
                log_r_max, r_max)
        parts = (
            checkpoint(_chunk_term, *args, use_reentrant=False)
            if grad_on
            else _chunk_term(*args)
        )
        m = mask[:, lo:hi]
        total = total + (parts[0] * m).sum()
        with torch.no_grad():
            sums["reverse"] += float((parts[1] * m).sum())
            sums["forward"] += float((parts[2] * m).sum())
            sums["r"] += float((parts[3] * m).sum())
            sums["r_sat"] += float(((parts[3] >= r_max).float() * m).sum())
        del parts

    loss = total / denom
    metrics = {
        "ratio_fwd_kl": float(loss.detach()),
        "ratio_fwd_tokens": float(denom),
        "ratio_fwd_reverse": sums["reverse"] / float(denom),
        "ratio_fwd_forward": sums["forward"] / float(denom),
        "ratio_fwd_r_mean": sums["r"] / float(denom),
        "ratio_fwd_r_saturated": sums["r_sat"] / float(denom),
    }
    return loss * coef, metrics
