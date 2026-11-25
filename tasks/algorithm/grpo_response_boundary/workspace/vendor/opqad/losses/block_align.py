"""Per-block hidden-state alignment between a quantized student and a frozen teacher.

Why this exists
---------------
The current MOPD objective is a *detached* top-k reward fed through a
REINFORCE-style estimator (``core_algos.compute_policy_loss_vanilla``, the 3-D
branch, which the code itself documents as "treating A as constant").  Because
the reward ``-w_v (log s_v - log t_v)`` is a function of the student's own
current log-probs, differentiating it as a constant drops a term:

    implemented:  dL   = sum_v w_v log(s_v/t_v) grad log s_v
    intended:     dJ   = dL + ((1-J)/M_k) * dM_k ,  M_k = sum_{v in top-k} s_v

so gradient descent on ``L`` both lowers the KL *and* raises ``M_k``, the mass
the student puts on its own current top-k.  That second term is a
self-reinforcing sharpening pressure with nothing opposing it (entropy_coeff=0,
use_kl_loss=false, use_kl_in_reward=false).

Block alignment sidesteps that failure mode entirely: it is a *direct*
differentiable loss on the student's activations, so there is no detached
reward and no missing term.  It is also a strictly better-conditioned signal for
KV-cache quantization QAT, because quantization error is injected at every
block's K/V and a logits-only objective observes it only after 28 layers of
mixing.

Metric choice
-------------
Raw MSE is the wrong default.  In a pre-norm transformer the hidden-state RMS
grows monotonically with depth (often by 1-2 orders of magnitude), so an
unweighted sum of per-layer MSEs is dominated by the last few blocks and the
early blocks -- where the quantization error actually enters -- contribute
almost nothing.  Both metrics here are scale-free:

  ``cosine``  1 - cos(h_s, h_t), per position, then masked-mean.
  ``nmse``    ||h_s - h_t||^2 / (||h_t||^2 + eps), per position, then masked-mean.
  ``oprd_vanilla`` L2-normalize both vectors per position, then compare them
                with squared error (the same-backbone OPRD objective).
  ``oprd_halfnorm`` divide both vectors by the square root of their L2 norm.

Everything is reduced in fp32 regardless of the input dtype.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "block_alignment_loss",
    "select_layers",
    "subsample_positions",
]

_EPS = 1e-6


def select_layers(num_layers: int, every: int = 1, include_last: bool = True) -> List[int]:
    """Indices of the blocks to align.

    Keeping all 28 blocks costs ``num_layers * B * T * D`` of activation memory
    for the student (which must survive until backward).  For the 7B profile at
    ``B=4, T=18432, D=3584`` that is 14.8 GB (13.8 GiB) in bf16, on top of an actor
    that already peaks near the device limit.  ``every=4`` keeps 8 of 28 blocks
    and cuts that to 4.2 GB.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    idx = list(range(0, num_layers, every))
    if include_last and (num_layers - 1) not in idx:
        idx.append(num_layers - 1)
    return idx


def subsample_positions(
    response_mask: torch.Tensor,
    keep_fraction: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Thin the response mask to ``keep_fraction`` of its valid positions.

    The alignment target is a per-position expectation, so an unbiased random
    subset estimates the same quantity at ``1/keep_fraction`` the activation
    cost.  Returns a mask with the same shape/dtype as ``response_mask``.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    if keep_fraction == 1.0:
        return response_mask
    valid = response_mask.bool()
    draw = torch.rand(valid.shape, device=valid.device, generator=generator)
    return (valid & (draw < keep_fraction)).to(response_mask.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` [B, T] over ``mask`` [B, T]; 0 when the mask is empty."""
    denom = mask.sum()
    if denom == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / denom


def block_alignment_loss(
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
    response_mask: torch.Tensor,
    *,
    metric: str = "cosine",
    layer_weights: Optional[Sequence[float]] = None,
    layer_scales: Optional[Sequence[float]] = None,
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align each student block output to the corresponding teacher block output.

    Args:
        student_hidden: ``L`` tensors of shape ``[B, T, D]``, differentiable.
        teacher_hidden: ``L`` tensors of shape ``[B, T, D]`` from the frozen
            teacher.  Detached internally, so callers need not.
        response_mask: ``[B, T]``; positions to score.  Prompt positions are
            excluded by the caller in the same way the OPD reward excludes them.
        metric: ``"cosine"``, ``"nmse"``, or ``"token_huber_layer_norm"``.
            The latter uses one median teacher-energy scale per layer and a
            Huber penalty on the normalized residual norm.
        layer_weights: optional per-layer weights; normalized to sum to 1.
            Defaults to uniform.

    Returns:
        ``(loss, metrics)`` where ``loss`` is a scalar and ``metrics`` maps
        ``"block_align/layer_<i>"`` to that block's mean distance.  The per-layer
        breakdown is the point: it shows *where* the quantized student diverges,
        which a logits-only objective cannot.
    """
    if len(student_hidden) != len(teacher_hidden):
        raise ValueError(
            f"student has {len(student_hidden)} blocks, teacher has {len(teacher_hidden)}"
        )
    if not student_hidden:
        raise ValueError("no blocks supplied")
    if metric not in ("cosine", "nmse", "oprd_vanilla", "oprd_halfnorm", "token_huber_layer_norm"):
        raise ValueError(f"unknown metric {metric!r}, expected cosine/nmse/oprd_vanilla/oprd_halfnorm/token_huber_layer_norm")
    if huber_delta <= 0:
        raise ValueError(f"huber_delta must be positive, got {huber_delta}")

    n = len(student_hidden)
    # The default objective is one mean over the effective layer × batch ×
    # token positions.  With one shared response mask and equal layer weight,
    # this is algebraically identical to averaging the per-layer masked means,
    # but spelling it out prevents a future layer-specific reduction from
    # silently changing the objective.  ``layer_weights`` remains as a
    # backwards-compatible opt-in for old experiments.
    direct_global_mean = layer_weights is None
    if layer_weights is None:
        weights = None
    else:
        if len(layer_weights) != n:
            raise ValueError(f"layer_weights has {len(layer_weights)} entries, expected {n}")
        total = float(sum(layer_weights))
        if total <= 0:
            raise ValueError("layer_weights must sum to a positive value")
        weights = [float(w) / total for w in layer_weights]

    mask = response_mask.to(dtype=torch.float32)
    if layer_scales is not None and len(layer_scales) != n:
        raise ValueError(f"layer_scales has {len(layer_scales)} entries, expected {n}")
    loss = None
    global_sum = None
    metrics: Dict[str, float] = {}

    for i, (hs, ht) in enumerate(zip(student_hidden, teacher_hidden)):
        if hs.shape != ht.shape:
            raise ValueError(f"block {i}: student {tuple(hs.shape)} != teacher {tuple(ht.shape)}")
        if hs.shape[:2] != mask.shape:
            raise ValueError(
                f"block {i}: hidden {tuple(hs.shape[:2])} does not match mask {tuple(mask.shape)}"
            )
        # Reductions in fp32: bf16 accumulation over D=3584 loses several bits
        # and these differences are small by construction.
        s = hs.float()
        t = ht.float().detach()

        if metric == "cosine":
            dist = 1.0 - torch.nn.functional.cosine_similarity(s, t, dim=-1, eps=_EPS)
        elif metric == "nmse":
            num = (s - t).pow(2).sum(dim=-1)
            den = t.pow(2).sum(dim=-1) + _EPS
            dist = num / den
        elif metric in ("oprd_vanilla", "oprd_halfnorm"):
            # OPRD: projection is identity for the same backbone.  Normalize
            # each response-token representation on both sides before MSE.
            power = 1.0 if metric == "oprd_vanilla" else 0.5
            s = s / s.norm(dim=-1, keepdim=True).clamp_min(_EPS).pow(power)
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(_EPS).pow(power)
            dist = (s - t).pow(2).sum(dim=-1)
        else:
            # One robust scale per layer, rather than one denominator per
            # token.  The median is computed over valid positions only.
            if layer_scales is None:
                valid_t = t[mask.to(torch.bool)]
                layer_energy = valid_t.square().sum(dim=-1)
                scale = torch.sqrt(layer_energy.median() + _EPS)
            else:
                scale = torch.as_tensor(
                    float(layer_scales[i]), device=t.device, dtype=torch.float32
                ).clamp_min(_EPS).sqrt()
            residual = (s - t).norm(dim=-1) / scale
            delta = float(huber_delta)
            abs_r = residual.abs()
            dist = torch.where(
                abs_r <= delta, 0.5 * residual.square(), delta * (abs_r - 0.5 * delta)
            )

        if not torch.isfinite(dist).all():
            raise FloatingPointError(f"non-finite block alignment distance at layer {i}, metric={metric}")
        per_layer = _masked_mean(dist, mask)
        metrics[f"block_align/layer_{i}"] = per_layer.detach().item()
        if direct_global_mean:
            contribution = (dist * mask).sum()
            global_sum = contribution if global_sum is None else global_sum + contribution
        else:
            term = per_layer * weights[i]
            loss = term if loss is None else loss + term

    if direct_global_mean:
        denom = mask.sum()
        loss = global_sum / (denom * n) if denom != 0 else global_sum * 0.0
    assert loss is not None
    metrics["block_align/loss"] = loss.detach().item()
    metrics["block_align/tokens"] = float(mask.sum().item())
    return loss, metrics
