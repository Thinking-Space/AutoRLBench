"""How closely does the training forward reproduce the rollout engine?

Why this metric exists
----------------------
The trainer already ships a ``rollout_corr/*`` metric family, but on this
configuration every one of them is identically zero, and not because anything
is healthy.  Two things conspire:

* ``bypass_old_logprob_for_rollout=true`` aliases ``old_log_probs`` to the
  rollout engine's log-probs, so no recomputation happens; then
* on-policy (``ppo_epochs=1``), the policy loss overwrites ``old_log_prob``
  with ``log_prob_for_loss.detach()`` -- the very tensor it just computed.

So ``actor/ppo_kl``, ``rollout_corr/kl``, ``log_ppl_diff`` and friends all
compare a tensor to itself.  They cannot report a problem.

What this measures instead is the comparison that matters for QAT: the actor's
recomputed log-prob of a sampled token versus the log-prob the *engine* assigned
when it sampled it.  With a TurboQuant rollout and a fake-quantized actor, that
gap is a running measurement of whether the training-time quantizer reproduces
the deployed one -- the same quantity
``scripts/parity_forward_logprobs.py`` measures offline, but on the real
training distribution and with no extra forward pass.

Reading the numbers
-------------------
``corr`` is the discriminating one.  Framework-versus-engine kernel differences
(bf16 reductions, different attention implementations) produce a small ``mae``
with ``corr`` very close to 1.  A quantizer mismatch decorrelates instead: the
two sides perturb *different* tokens.  ``mae`` alone cannot tell those apart.
"""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["rollout_logprob_agreement"]


@torch.no_grad()
def rollout_logprob_agreement(
    actor_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str = "qat",
) -> Dict[str, float]:
    """Masked agreement statistics between two ``[B, T]`` log-prob tensors.

    Returns an empty dict when there is nothing to measure, so the caller can
    always ``update()`` with the result.
    """
    if actor_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"shape mismatch: actor {tuple(actor_log_prob.shape)} vs rollout "
            f"{tuple(rollout_log_prob.shape)}"
        )
    if response_mask.shape != actor_log_prob.shape:
        raise ValueError(
            f"response_mask {tuple(response_mask.shape)} does not match "
            f"log-probs {tuple(actor_log_prob.shape)}"
        )

    a = actor_log_prob.float()
    b = rollout_log_prob.to(a.dtype)
    m = response_mask.to(a.dtype)
    total = m.sum()
    # Correlation is undefined for a single point, and a zero-token micro-batch
    # would divide by zero.
    if total.item() <= 1:
        return {}

    diff = (a - b) * m
    mean_a = (a * m).sum() / total
    mean_b = (b * m).sum() / total
    da, db = (a - mean_a) * m, (b - mean_b) * m
    denom = torch.sqrt((da * da).sum() * (db * db).sum())
    corr = (da * db).sum() / denom if denom > 0 else torch.zeros((), device=a.device)

    return {
        f"{prefix}/rollout_logprob_mae": (diff.abs().sum() / total).item(),
        f"{prefix}/rollout_logprob_bias": (diff.sum() / total).item(),
        f"{prefix}/rollout_logprob_corr": corr.item(),
        f"{prefix}/rollout_logprob_frac_gt_0p1": (
            ((diff.abs() > 0.1).to(a.dtype) * m).sum() / total
        ).item(),
    }
