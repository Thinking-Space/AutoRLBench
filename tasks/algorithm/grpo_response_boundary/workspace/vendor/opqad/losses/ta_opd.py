"""TA-OPD: the tail the top-k OPD objective cannot see.

The problem, in one line
------------------------
The top-k reward is a function only of the student's probabilities *inside* the
candidate set, renormalised.  Nothing in it refers to how much total mass sits
there.  So ``M = sum_{v in S} p_v`` is a free variable, and on the OSCAR 1.5B
arms it ran

    step        1      10      20
    M      0.8896  0.8036  0.5225      (val-topk/student_p_sum_intersection)
    M_T    0.8295  0.8145  0.7469      (val-topk/teacher_p_sum_intersection)

while the reward and ``overlap_ratio`` improved monotonically throughout.  By
step 20 nearly half the student's probability lived in a region the objective
cannot see, and ``training_log_ppl`` had risen 3.1 nats.

It is worse than blindness.  The live reward is
``r_k = softmax_K(s)_k (t_k - s_k)``: the top-k probabilities renormalised by M
applied to ABSOLUTE log-ratios, which is algebraically

    R  =  -D_KL(pbar || qbar)  +  log M_T  -  log M

verified numerically to 8.9e-16.  The reward is maximised, so the ``-log M``
term pays for reducing M.  Proposition 1 of arXiv 2608.14728 says the
normalised top-k loss gives the tail zero gradient; ours gives it a gradient
pointing the wrong way.

The term
--------
TA-OPD (arXiv 2608.14728, Eq. 4) partitions the vocabulary into the teacher's
top-k plus one virtual "everything else" token and takes the KL on that
(k+1)-point partition::

    l_TA = sum_{v in S} p(v) log(p(v)/q(v))  +  p_tail log(p_tail/q_tail)

with ``S = TopK(q, k)`` and ``p(v)`` ABSOLUTE -- no division by their own sum.
The division is the whole defect: under renormalisation the objective is a
function of z only through the softmax restricted to S, so every logit outside
S has identically zero gradient and the in-set gradients sum to zero.  Both
halves of that are checked by autograd in tests/test_ta_opd.py rather than
retyped from the paper.

The tail gradient this restores is, for v outside S::

    d l_TA / d z_v  =  p_v (r_T - l_TA),     r_T = log(p_T / q_T)

so ``r_T`` is the sign-determining quantity: the student's tail heavier than
the teacher's means descent pushes tail logits down.  It is logged as
``actor/ta_opd_r_T``.

Cost
----
Both inputs are tensors the OPD path already has: ``topk_log_probs``
(differentiable, in the graph) and the teacher's log-probs on the same ids
(shipped by the teacher worker).  Both are ABSOLUTE log-probabilities --
``logprobs_for_ids_fp32`` normalises over the whole row and chunks only the
FP32 cast -- which is the precondition this term needs.  A renormalised top-k
interface could not express it at all.

No teacher hidden state, no resident ``lm_head``, no [B,T,V] tensor: two
logsumexps and one elementwise product over [B,T,K] on top of what is already
computed, against the full-vocabulary KL's teacher hidden state (0.19 GB), head
(0.47 GB) and chunked log-softmax over 151936.

Numerics
--------
``M`` reaches 0.9997 on k2v3, where ``1 - M`` in float32 keeps only ~3 digits.
Everything here goes through ``logsumexp`` and ``log(-expm1(log M))`` so the
tail mass is computed in log space and never as a cancelling subtraction.
"""


from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = ["ta_opd_loss"]

# log M is <= 0.  Clamp it just below 0 so log(-expm1(log M)) stays finite when
# the top-k captures the entire distribution to within float32.
_LOG_M_MAX = -1e-7


def _log_mass(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log M, log(1-M)) from per-candidate absolute log-probabilities.

    ``logp`` is ``[..., K]``.  Non-finite candidates are dropped rather than
    poisoning the sum: the OPD path pads short candidate rows with -inf, and a
    NaN here would silently zero the whole term through the loss.
    """
    logp = torch.where(torch.isfinite(logp), logp, torch.full_like(logp, -float("inf")))
    log_m = torch.logsumexp(logp.float(), dim=-1).clamp(max=_LOG_M_MAX)
    # log(1 - M) = log(-(exp(log M) - 1)) = log(-expm1(log M)), stable as M->1.
    log_tail = torch.log(-torch.expm1(log_m))
    return log_m, log_tail


def ta_opd_loss(
    student_topk_logp: torch.Tensor,
    teacher_topk_logp: torch.Tensor,
    response_mask: torch.Tensor,
    coef: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """TA-OPD (arXiv 2608.14728, Eq. 4): reverse KL on top-k PLUS a tail token.

        l_TA = sum_{v in S} p(v) log(p(v)/q(v))  +  p_tail log(p_tail/q_tail)

    with ``S = TopK(q, k)`` -- the TEACHER's top-k -- ``p`` the student, ``q``
    the teacher, and ``p_tail = 1 - sum_{v in S} p(v)``.  The top-k
    probabilities are **absolute**: there is no division by their own sum.
    That is the whole difference from the objective the paper criticises.

    Relation to what this repository runs
    -------------------------------------
    The live reward is ``r_k = softmax_K(s)_k (t_k - s_k)``, i.e. the top-k
    probabilities renormalised by M and applied to absolute log-ratios.
    Algebraically that equals ``-D_KL(pbar||qbar) + log M_T - log M``: the
    paper's criticised baseline plus an explicit ``-log M`` term.  Since the
    reward is maximised, that term pays for reducing M -- it is worse than the
    "zero tail gradient" Proposition 1 describes, not merely equal to it.

    A caveat that must travel with any comparison to Proposition 1: our reward
    is **detached** and enters a policy gradient, so the proposition -- which is
    about differentiating l_norm -- does not describe our update literally.  Our
    update does move the tail (``g_v = -A_bar p_v`` off S), it just moves it in
    a direction carrying no information about the teacher's tail.  This loss,
    by contrast, is differentiated directly, so the proposition's arithmetic
    applies to it as written.

    Bounds
    ------
    The 2-point coarsening ``{inside S, outside S}`` -- which is all a pure
    mass constraint sees -- is below this, and this is below the full
    vocabulary, by the data-processing inequality::

        L_2  <=  l_TA  <=  D_KL(p || q)

    Both inequalities are checked over five seeds in tests/test_ta_opd.py.
    L_2 was run as its own loss term for a while; it is gone because l_TA
    dominates it at the same cost.

    Args:
        student_topk_logp: ``[B, T, K]`` student log-probs on the candidate ids,
            differentiable and ABSOLUTE.  With ``TOP_K_STRATEGY=only_tch`` these
            are the student's log-probs on the teacher's top-k, which is what
            the paper's S requires.
        teacher_topk_logp: ``[B, T, K]`` frozen teacher log-probs on the SAME
            ids, also absolute.
        response_mask: ``[B, T]``.
        coef: linear scale, already applied to the returned loss.

    Returns:
        ``(loss, metrics)``, token-mean aggregated.
    """
    if student_topk_logp.shape != teacher_topk_logp.shape:
        raise ValueError(
            f"student {tuple(student_topk_logp.shape)} and teacher "
            f"{tuple(teacher_topk_logp.shape)} candidate log-probs must have the "
            "same shape -- TA-OPD's partition is one partition"
        )
    if student_topk_logp.dim() != 3:
        raise ValueError(
            f"expected [B, T, K] candidate log-probs, got {tuple(student_topk_logp.shape)}"
        )
    if response_mask.shape != student_topk_logp.shape[:2]:
        raise ValueError(
            f"response_mask must be {student_topk_logp.shape[:2]}, got {tuple(response_mask.shape)}"
        )

    s = torch.where(
        torch.isfinite(student_topk_logp),
        student_topk_logp,
        torch.full_like(student_topk_logp, -float("inf")),
    ).float()
    with torch.no_grad():
        t = torch.where(
            torch.isfinite(teacher_topk_logp),
            teacher_topk_logp,
            torch.full_like(teacher_topk_logp, -float("inf")),
        ).float()

    # Head: absolute p(v), no renormalisation.  -inf candidates contribute 0
    # because exp(-inf) = 0, but (-inf) - (-inf) is NaN, so the difference is
    # taken only where both are finite.
    both = torch.isfinite(s) & torch.isfinite(t)
    ratio = torch.where(both, s - t, torch.zeros_like(s))
    head = (s.exp() * ratio).sum(dim=-1)

    log_m, log_tail = _log_mass(s)
    with torch.no_grad():
        log_mt, log_tail_t = _log_mass(t)
    tail = log_tail.exp() * (log_tail - log_tail_t)

    per_token = head + tail
    mask = response_mask.to(per_token.dtype)
    denom = mask.sum()
    if float(denom) == 0.0:
        # Every key, not just two.  The caller iterates a fixed key list and
        # indexes the dict directly, so a short dict here is a KeyError ten
        # hours into a run rather than a missing chart.  Zeros rather than NaN:
        # these are averaged across micro-batches and a NaN would poison the
        # step's whole row.
        zero = student_topk_logp.sum() * 0.0
        return zero, {
            "ta_opd": 0.0,
            "ta_opd_tokens": 0.0,
            "ta_opd_head": 0.0,
            "ta_opd_tail": 0.0,
            "ta_opd_r_T": 0.0,
            "ta_opd_c": 0.0,
            "ta_opd_p_tail": 0.0,
            "ta_opd_q_tail": 0.0,
            "ta_opd_M": 0.0,
            "ta_opd_M_teacher": 0.0,
        }

    loss = (per_token * mask).sum() / denom
    with torch.no_grad():
        # r_T = log(p_T/q_T) is the sign-determining factor of the tail
        # gradient: dl_TA/dz_v = p_v (r_T - l_TA) for v outside S.  Positive
        # means the student's tail is heavier than the teacher's, so descent
        # pushes tail logits DOWN -- which is the direction the collapse needs
        # and the normalised objective cannot produce.
        r_t = log_tail - log_tail_t
        # c compares in/out odds; -sign(c) is the direction M is driven, and
        # c = 0 exactly when M = M_T, and -sign(c) is the direction M is
        # driven by any term that constrains the head/tail split.
        c = (log_m - log_tail) - (log_mt - log_tail_t)
        metrics = {
            "ta_opd": float(loss.detach()),
            "ta_opd_tokens": float(denom),
            "ta_opd_head": float(((head * mask).sum() / denom).detach()),
            "ta_opd_tail": float((tail * mask).sum() / denom),
            "ta_opd_r_T": float((r_t * mask).sum() / denom),
            "ta_opd_c": float((c * mask).sum() / denom),
            "ta_opd_p_tail": float((log_tail.exp() * mask).sum() / denom),
            "ta_opd_q_tail": float((log_tail_t.exp() * mask).sum() / denom),
            "ta_opd_M": float((log_m.exp() * mask).sum() / denom),
            "ta_opd_M_teacher": float((log_mt.exp() * mask).sum() / denom),
        }
    return loss * coef, metrics
