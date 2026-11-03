"""Length-weighted policy objective with token-level importance ratios."""

import torch

from verl.trainer.ppo.core_algos import register_policy_loss


@register_policy_loss("luspo_token")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="seq-mean-token-sum",
    config=None,
    rollout_is_weights=None,
):
    """Return the policy objective and masked diagnostic metrics.

    The outcome advantage is constant within each response. The actor passes
    token-shaped advantages, so recover the response-level value first.
    """
    if config is None:
        raise ValueError("Actor configuration is required")
    mask = response_mask.to(log_prob.dtype)
    lengths = mask.sum(-1, keepdim=True)
    sequence_advantage = (advantages * mask).sum(-1, keepdim=True) / lengths.clamp_min(1)
    delta = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = delta.exp()
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    direct = -sequence_advantage * ratio
    clipped = -sequence_advantage * ratio.clamp(1 - clip_low, 1 + clip_high)
    per_token_loss = torch.maximum(direct, clipped)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = (per_token_loss * lengths).mean()

    denominator = mask.sum().clamp_min(1)
    metrics = {
        "actor/pg_clipfrac": (((clipped > direct).to(mask.dtype) * mask).sum() / denominator).detach().item(),
        "actor/ppo_kl": ((-delta * mask).sum() / denominator).detach().item(),
        "actor/padding_fraction": (1 - mask.mean()).detach().item(),
        "actor/luspo_token_loss": loss.detach().item(),
    }
    return loss, metrics
