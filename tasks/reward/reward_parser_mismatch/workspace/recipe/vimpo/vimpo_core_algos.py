import math

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    max_per_token = None
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_max = logits[:, start:end].max(dim=-1, keepdim=True).values
        max_per_token = chunk_max if max_per_token is None else torch.maximum(max_per_token, chunk_max)

    sum_exp = torch.zeros_like(max_per_token, dtype=torch.float32)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        sum_exp = sum_exp + torch.exp(logits[:, start:end] - max_per_token).sum(
            dim=-1,
            keepdim=True,
            dtype=torch.float32,
        )
    return max_per_token.to(torch.float32) + torch.log(sum_exp)


def build_vimpo_token_term(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Build the per-token VIMPO term beta * (log pi/ref - KL)."""
    response_mask = response_mask.to(dtype=log_prob.dtype)
    beta = torch.as_tensor(beta, dtype=log_prob.dtype, device=log_prob.device)
    return beta * (log_prob - ref_log_prob - token_kl) * response_mask


def _masked_sequence_sum(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    if values.shape != response_mask.shape:
        raise ValueError(f"{values.shape=} does not match {response_mask.shape=}")
    response_mask = response_mask.to(dtype=values.dtype)
    return torch.sum(values * response_mask, dim=-1)


def masked_whiten_distributed(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    process_group=None,
) -> torch.Tensor:
    """Whiten valid entries using stats reduced across distributed ranks.

    Falls back to local masked whitening when torch.distributed is unavailable,
    uninitialized, or has a single rank. The returned tensor has zero mean and
    unit variance over all valid entries in the distributed process group.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return verl_F.masked_whiten(values, mask)

    world_size = torch.distributed.get_world_size(group=process_group)
    if world_size <= 1:
        return verl_F.masked_whiten(values, mask)

    valid = mask.to(dtype=torch.float64, device=values.device)
    value64 = values.to(dtype=torch.float64)
    stats = torch.stack(
        [
            valid.sum(),
            (value64 * valid).sum(),
            (value64.square() * valid).sum(),
        ]
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=process_group)

    count = stats[0].clamp_min(1.0)
    mean = stats[1] / count
    var = (stats[2] / count - mean.square()).clamp_min(0.0)
    whitened = (value64 - mean) * torch.rsqrt(var + eps)
    return whitened.to(dtype=values.dtype)


def _group_mean_by_uid(values: torch.Tensor, uids) -> torch.Tensor:
    uid_list = list(uids)
    if len(uid_list) != values.shape[0]:
        raise ValueError(f"Expected {values.shape[0]} uid values, got {len(uid_list)}.")

    groups: dict[object, list[int]] = {}
    for idx, uid in enumerate(uid_list):
        key = uid.item() if hasattr(uid, "item") else uid
        groups.setdefault(key, []).append(idx)

    baseline = torch.empty_like(values)
    for indices in groups.values():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=values.device)
        baseline[index_tensor] = values[index_tensor].mean()
    return baseline


def compute_grouped_baseline_by_uid(values: torch.Tensor, uids) -> torch.Tensor:
    """Compute a detached per-sample baseline from values grouped by prompt uid."""
    if values.ndim != 1:
        raise ValueError(f"Expected 1D values, got {values.shape=}")
    return _group_mean_by_uid(values, uids).detach()


def compute_reward_baseline_by_uid(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uids,
    cap_v0: float | None = None,
) -> torch.Tensor:
    """Compute a detached per-sequence reward baseline grouped by prompt uid."""
    final_reward = _masked_sequence_sum(token_level_rewards, response_mask)
    baseline = compute_grouped_baseline_by_uid(final_reward, uids)
    if cap_v0 is not None:
        baseline = baseline.clamp_max(torch.as_tensor(cap_v0, dtype=baseline.dtype, device=baseline.device))
    return baseline


@torch.no_grad()
def compute_selected_logprob_logit_l2_factor(
    valid_response_logits: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute ||d log p(y) / d logits||_2 for valid sampled response tokens."""
    valid_response_mask = response_mask.to(torch.bool)
    valid_log_prob = torch.masked_select(log_prob, valid_response_mask).detach().to(torch.float32)
    logits = valid_response_logits.detach().to(torch.float32)
    log_z = torch.logsumexp(logits, dim=-1)
    sum_pi_squared = torch.zeros_like(log_z)
    chunk_size = max(1, int(vocab_chunk_size))
    for start in range(0, logits.shape[-1], chunk_size):
        end = min(start + chunk_size, logits.shape[-1])
        sum_pi_squared = sum_pi_squared + torch.exp(2.0 * (logits[:, start:end] - log_z.unsqueeze(-1))).sum(dim=-1)
    selected_prob = torch.exp(valid_log_prob)
    scale_squared = (1.0 - 2.0 * selected_prob + sum_pi_squared).clamp_min(0.0)
    return torch.sqrt(scale_squared)


@torch.no_grad()
def compute_vimpo_value_logit_grad_rms(
    valid_response_logits: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_kl: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    reward_baseline: torch.Tensor,
    beta: float,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor | None:
    valid_response_mask = response_mask.to(torch.bool)
    if not valid_response_mask.any():
        return None

    dtype = log_prob.dtype
    response_mask_f = response_mask.to(dtype=dtype)
    beta_tensor = torch.as_tensor(beta, dtype=dtype, device=log_prob.device)
    vimpo_token_term = beta_tensor * (log_prob - ref_log_prob - token_kl) * response_mask_f
    terminal_value = torch.sum(vimpo_token_term, dim=-1)
    final_reward = torch.sum(token_level_rewards.to(dtype) * response_mask_f, dim=-1)
    terminal_target = final_reward - reward_baseline.to(dtype=dtype, device=log_prob.device)
    residual = terminal_value - terminal_target

    if value_loss_type == "huber":
        huber_delta_tensor = torch.as_tensor(huber_delta, dtype=dtype, device=log_prob.device)
        residual_grad = residual.clamp(min=-float(huber_delta_tensor.item()), max=float(huber_delta_tensor.item()))
    elif value_loss_type == "squared_error":
        residual_grad = residual
    else:
        raise ValueError(f"vimpo_value_loss_type must be 'squared_error' or 'huber', got {value_loss_type!r}.")

    batch_size = max(1, log_prob.shape[0])
    coeff = beta_tensor * response_mask_f * residual_grad.unsqueeze(-1) / float(batch_size)
    valid_coeff = torch.masked_select(coeff, valid_response_mask).detach().to(torch.float32)
    logit_factor = compute_selected_logprob_logit_l2_factor(
        valid_response_logits=valid_response_logits,
        log_prob=log_prob,
        response_mask=response_mask,
        vocab_chunk_size=vocab_chunk_size,
    )
    value_scales = valid_coeff.abs() * logit_factor
    return torch.sqrt(torch.mean(value_scales.square()))


@torch.no_grad()
def compute_vimpo_value_logit_grad_stats(
    valid_response_logits: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_kl: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    reward_baseline: torch.Tensor,
    beta: float,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
    vocab_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not response_mask.to(torch.bool).any():
        zero = torch.zeros((), dtype=torch.float32, device=log_prob.device)
        return zero, zero
    value_rms = compute_vimpo_value_logit_grad_rms(
        valid_response_logits=valid_response_logits,
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        token_kl=token_kl,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        reward_baseline=reward_baseline,
        beta=beta,
        value_loss_type=value_loss_type,
        huber_delta=huber_delta,
        vocab_chunk_size=vocab_chunk_size,
    )
    if value_rms is None:
        return None
    count = response_mask.to(torch.bool).sum().to(dtype=torch.float32, device=value_rms.device)
    return value_rms.to(torch.float32).square() * count, count


@torch.no_grad()
def compute_vimpo_ppo_actor_logit_grad_rms(
    valid_response_logits: torch.Tensor,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    vimpo_actor_coeff: float,
    token_mean_coeff_scale: torch.Tensor | float,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
    rollout_is_weights: torch.Tensor | None = None,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor | None:
    valid_response_mask = response_mask.to(torch.bool)
    if not valid_response_mask.any():
        return None

    negative_approx_kl_raw = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl_raw, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c

    active_unclipped = pg_losses1 >= pg_losses2
    active_unclipped = torch.where(
        advantages < 0,
        active_unclipped & (clip_pg_losses1 <= pg_losses3),
        active_unclipped,
    )
    active_unclipped = active_unclipped & (negative_approx_kl_raw > -20.0) & (negative_approx_kl_raw < 20.0)
    active_unclipped = active_unclipped.to(dtype=log_prob.dtype) * response_mask.to(dtype=log_prob.dtype)

    coeff_scale = torch.as_tensor(token_mean_coeff_scale, dtype=log_prob.dtype, device=log_prob.device)
    coeff = -float(vimpo_actor_coeff) * coeff_scale * ratio * advantages * active_unclipped
    if rollout_is_weights is not None:
        coeff = coeff * rollout_is_weights

    valid_coeff = torch.masked_select(coeff, valid_response_mask).detach().to(torch.float32)
    logit_factor = compute_selected_logprob_logit_l2_factor(
        valid_response_logits=valid_response_logits,
        log_prob=log_prob,
        response_mask=response_mask,
        vocab_chunk_size=vocab_chunk_size,
    )
    actor_scales = valid_coeff.abs() * logit_factor
    return torch.sqrt(torch.mean(actor_scales.square()))


@torch.no_grad()
def compute_vimpo_ppo_actor_logit_grad_stats(
    valid_response_logits: torch.Tensor,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    vimpo_actor_coeff: float,
    token_mean_coeff_scale: torch.Tensor | float,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
    rollout_is_weights: torch.Tensor | None = None,
    vocab_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not response_mask.to(torch.bool).any():
        zero = torch.zeros((), dtype=torch.float32, device=log_prob.device)
        return zero, zero
    actor_rms = compute_vimpo_ppo_actor_logit_grad_rms(
        valid_response_logits=valid_response_logits,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        vimpo_actor_coeff=vimpo_actor_coeff,
        token_mean_coeff_scale=token_mean_coeff_scale,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        clip_ratio_c=clip_ratio_c,
        rollout_is_weights=rollout_is_weights,
        vocab_chunk_size=vocab_chunk_size,
    )
    if actor_rms is None:
        return None
    count = response_mask.to(torch.bool).sum().to(dtype=torch.float32, device=actor_rms.device)
    return actor_rms.to(torch.float32).square() * count, count


@torch.no_grad()
def compute_vimpo_adaptive_actor_coeff(
    value_logit_grad_rms: torch.Tensor | float,
    actor_logit_grad_rms_unscaled: torch.Tensor | float,
    value_logit_grad_rms_ema: float,
    actor_logit_grad_rms_unscaled_ema: float,
    alpha: float,
    ema_decay: float,
    eps: float,
) -> tuple[float, float, float]:
    """Update zero-start EMA state and return a detached adaptive actor coefficient."""
    if not 0 <= ema_decay < 1:
        raise ValueError(f"vimpo_adaptive_actor_ema_decay must be in [0, 1), got {ema_decay}.")
    if eps <= 0:
        raise ValueError(f"vimpo_adaptive_actor_eps must be positive, got {eps}.")

    value_rms = float(torch.as_tensor(value_logit_grad_rms).detach().to(torch.float32).item())
    actor_rms = float(torch.as_tensor(actor_logit_grad_rms_unscaled).detach().to(torch.float32).item())
    new_value_ema = ema_decay * float(value_logit_grad_rms_ema) + (1.0 - ema_decay) * value_rms
    new_actor_ema = ema_decay * float(actor_logit_grad_rms_unscaled_ema) + (1.0 - ema_decay) * actor_rms
    coeff = float(alpha) * new_value_ema / (new_actor_ema + float(eps))
    return coeff, new_value_ema, new_actor_ema


def _compute_terminal_value_loss_stats(
    prediction: torch.Tensor,
    target: torch.Tensor,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError(f"{prediction.shape=} does not match {target.shape=}")

    if value_loss_type not in {"squared_error", "huber"}:
        raise ValueError(f"vimpo_value_loss_type must be 'squared_error' or 'huber', got {value_loss_type!r}.")
    if huber_delta <= 0:
        raise ValueError(f"vimpo_value_huber_delta must be positive, got {huber_delta}.")

    residual = prediction - target
    abs_residual = residual.abs()
    squared_error = residual.square()
    half_squared_error = 0.5 * squared_error
    huber_delta_tensor = torch.as_tensor(huber_delta, dtype=prediction.dtype, device=prediction.device)
    huber_loss = torch.where(
        abs_residual <= huber_delta_tensor,
        half_squared_error,
        huber_delta_tensor * (abs_residual - 0.5 * huber_delta_tensor),
    )
    per_sequence_loss = half_squared_error if value_loss_type == "squared_error" else huber_loss
    loss = per_sequence_loss.mean()
    terminal_rmse = torch.sqrt(torch.mean(residual.square()))
    target_rms = torch.sqrt(torch.mean(target.square()))
    return loss, {
        "terminal_rmse": terminal_rmse,
        "target_rms": target_rms,
        "terminal_rmse_over_target_rms": terminal_rmse / (target_rms + 1e-8),
        "terminal_squared_error_mean": squared_error.mean(),
        "terminal_half_squared_error_mean": half_squared_error.mean(),
        "terminal_huber_loss_mean": huber_loss.mean(),
        "terminal_huber_clipped_frac": (abs_residual > huber_delta_tensor).to(prediction.dtype).mean(),
        "terminal_abs_error_mean": abs_residual.mean(),
        "value_loss_type_huber": torch.as_tensor(
            float(value_loss_type == "huber"), dtype=prediction.dtype, device=prediction.device
        ),
        "value_huber_delta": huber_delta_tensor,
    }


def compute_value_loss_norm_denom(target: torch.Tensor) -> torch.Tensor:
    """Compute sequence-level target MSE denominator for normalized value loss."""
    return target.detach().square().mean()


def _scale_value_loss(
    value_loss: torch.Tensor,
    value_loss_normalized: bool,
    value_loss_coeff: float,
    value_loss_norm_denom: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not math.isfinite(value_loss_coeff) or value_loss_coeff < 0:
        raise ValueError(f"vimpo_value_loss_coeff must be non-negative, got {value_loss_coeff}.")

    coeff = torch.as_tensor(value_loss_coeff, dtype=value_loss.dtype, device=value_loss.device)
    normalized = torch.as_tensor(float(value_loss_normalized), dtype=value_loss.dtype, device=value_loss.device)
    metrics = {
        "value_loss_raw": value_loss.detach(),
        "value_loss_coeff": coeff.detach(),
        "value_loss_normalized": normalized.detach(),
    }
    scaled_loss = coeff * value_loss
    if value_loss_normalized:
        if value_loss_norm_denom is None:
            raise ValueError("vimpo_value_loss_normalized=True requires a PPO-mini-batch value loss denominator.")
        denom = torch.as_tensor(value_loss_norm_denom, dtype=value_loss.dtype, device=value_loss.device).detach()
        if denom.item() < 1e-4:
            raise ValueError(
                "vimpo_value_loss_normalized=True requires target MSE denominator >= 1e-4, "
                f"got {denom.item()}."
            )
        scaled_loss = scaled_loss / denom
        metrics["value_loss_norm_denom"] = denom.detach()
    metrics["value_loss_for_backward"] = scaled_loss.detach()
    return scaled_loss, metrics


def compute_vimpo_terminal_mse_loss(
    vimpo_token_term: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    reward_baseline: torch.Tensor | float | None = None,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
    value_loss_normalized: bool = False,
    value_loss_coeff: float = 1.0,
    value_loss_norm_denom: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the terminal-MSE VIMPO loss for final-reward-only training.

    The constructed terminal value is
        sum_k gamma^(k-T) * vimpo_token_term_k
    and the target is
        R_final - gamma^(-T) * mean(R_final).
    """
    if vimpo_token_term.shape != token_level_rewards.shape:
        raise ValueError(f"{vimpo_token_term.shape=} does not match {token_level_rewards.shape=}")
    if vimpo_token_term.shape != response_mask.shape:
        raise ValueError(f"{vimpo_token_term.shape=} does not match {response_mask.shape=}")
    if gamma <= 0:
        raise ValueError(f"Expected gamma > 0, got {gamma}")

    dtype = vimpo_token_term.dtype
    device = vimpo_token_term.device
    response_mask = response_mask.to(dtype=dtype)
    seq_lengths = response_mask.sum(dim=-1)
    seq_lengths_clamped = seq_lengths.clamp(min=1.0)

    positions = torch.arange(vimpo_token_term.shape[-1], device=device, dtype=dtype).unsqueeze(0)
    gamma_tensor = torch.as_tensor(gamma, device=device, dtype=dtype)

    if gamma == 1.0:
        discounted_weights = response_mask
        gamma_inv_length = torch.ones_like(seq_lengths_clamped)
    else:
        discounted_weights = torch.pow(gamma_tensor, positions - seq_lengths_clamped.unsqueeze(-1)) * response_mask
        gamma_inv_length = torch.pow(gamma_tensor, -seq_lengths_clamped)

    terminal_value = torch.sum(vimpo_token_term * discounted_weights, dim=-1)
    final_reward = _masked_sequence_sum(token_level_rewards.to(dtype), response_mask)
    if reward_baseline is None:
        reward_baseline = final_reward.mean()
    else:
        reward_baseline = torch.as_tensor(reward_baseline, dtype=dtype, device=device)
    terminal_target = final_reward - gamma_inv_length * reward_baseline

    raw_loss, terminal_stats = _compute_terminal_value_loss_stats(
        terminal_value,
        terminal_target,
        value_loss_type=value_loss_type,
        huber_delta=huber_delta,
    )
    loss, value_loss_metrics = _scale_value_loss(
        raw_loss,
        value_loss_normalized=value_loss_normalized,
        value_loss_coeff=value_loss_coeff,
        value_loss_norm_denom=value_loss_norm_denom,
    )
    metrics = {
        "terminal_value": terminal_value.mean(),
        "terminal_target": terminal_target.mean(),
        "final_reward": final_reward.mean(),
        "reward_baseline": reward_baseline.mean(),
        "value_loss_for_backward": loss.detach(),
        "terminal_rmse_over_target_rms": terminal_stats["terminal_rmse_over_target_rms"],
        "terminal_squared_error_mean": terminal_stats["terminal_squared_error_mean"],
        "terminal_half_squared_error_mean": terminal_stats["terminal_half_squared_error_mean"],
        "terminal_huber_loss_mean": terminal_stats["terminal_huber_loss_mean"],
        "terminal_huber_clipped_frac": terminal_stats["terminal_huber_clipped_frac"],
        "terminal_abs_error_mean": terminal_stats["terminal_abs_error_mean"],
        "value_loss_type_huber": terminal_stats["value_loss_type_huber"],
        "value_huber_delta": terminal_stats["value_huber_delta"],
        **value_loss_metrics,
    }
    return loss, metrics


def compute_vimpo_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_kl: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    gamma: float,
    reward_baseline: torch.Tensor | float | None,
    loss_agg_mode: str,
    global_batch_info: dict,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
    value_loss_normalized: bool = False,
    value_loss_coeff: float = 1.0,
    value_loss_norm_denom: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the VIMPO terminal-MSE loss and supporting metrics."""
    vimpo_kl_agg = agg_loss(
        loss_mat=token_kl,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **global_batch_info,
    )
    vimpo_token_term = build_vimpo_token_term(
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        token_kl=token_kl,
        response_mask=response_mask,
        beta=beta,
    )
    vimpo_terminal_mse, vimpo_terminal_metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=vimpo_token_term,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        gamma=gamma,
        reward_baseline=reward_baseline,
        value_loss_type=value_loss_type,
        huber_delta=huber_delta,
        value_loss_normalized=value_loss_normalized,
        value_loss_coeff=value_loss_coeff,
        value_loss_norm_denom=value_loss_norm_denom,
    )
    metrics = {
        "vimpo_kl": vimpo_kl_agg.detach(),
        "vimpo_terminal_mse": vimpo_terminal_mse.detach(),
        "vimpo_value_loss_raw": vimpo_terminal_metrics["value_loss_raw"].detach(),
        "vimpo_value_loss_for_backward": vimpo_terminal_metrics["value_loss_for_backward"].detach(),
        "vimpo_value_loss_coeff": vimpo_terminal_metrics["value_loss_coeff"].detach(),
        "vimpo_value_loss_normalized": vimpo_terminal_metrics["value_loss_normalized"].detach(),
        "vimpo_terminal_value": vimpo_terminal_metrics["terminal_value"].detach(),
        "vimpo_terminal_target": vimpo_terminal_metrics["terminal_target"].detach(),
        "vimpo_final_reward": vimpo_terminal_metrics["final_reward"].detach(),
        "vimpo_reward_baseline": vimpo_terminal_metrics["reward_baseline"].detach(),
        "terminal_rmse_over_target_rms": vimpo_terminal_metrics["terminal_rmse_over_target_rms"].detach(),
        "vimpo_value_loss_type_huber": vimpo_terminal_metrics["value_loss_type_huber"].detach(),
        "vimpo_value_huber_delta": vimpo_terminal_metrics["value_huber_delta"].detach(),
        "vimpo_terminal_squared_error_mean": vimpo_terminal_metrics["terminal_squared_error_mean"].detach(),
        "vimpo_terminal_huber_loss_mean": vimpo_terminal_metrics["terminal_huber_loss_mean"].detach(),
        "vimpo_terminal_huber_clipped_frac": vimpo_terminal_metrics["terminal_huber_clipped_frac"].detach(),
        "vimpo_terminal_abs_error_mean": vimpo_terminal_metrics["terminal_abs_error_mean"].detach(),
    }
    if "value_loss_norm_denom" in vimpo_terminal_metrics:
        metrics["vimpo_value_loss_norm_denom"] = vimpo_terminal_metrics["value_loss_norm_denom"].detach()
    return vimpo_terminal_mse, vimpo_kl_agg, metrics


def compute_soft_value_baseline_by_uid(
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    old_token_kl: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    uids,
    cap_v0: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build detached per-sequence soft V0 baselines grouped by prompt uid.

    The baseline estimates E[R - beta * sum_t KL_t | prompt] from the old
    rollout policy. For sampled estimators, ``old_token_kl`` is the sampled
    per-token KL estimate; for exact/candidate estimators it is the no-grad
    expected KL under the old policy.
    """
    if old_log_prob.shape != ref_log_prob.shape:
        raise ValueError(f"{old_log_prob.shape=} does not match {ref_log_prob.shape=}")
    if old_log_prob.shape != old_token_kl.shape:
        raise ValueError(f"{old_log_prob.shape=} does not match {old_token_kl.shape=}")
    if old_log_prob.shape != token_level_rewards.shape:
        raise ValueError(f"{old_log_prob.shape=} does not match {token_level_rewards.shape=}")
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(f"{old_log_prob.shape=} does not match {response_mask.shape=}")

    uid_list = list(uids)
    if len(uid_list) != old_log_prob.shape[0]:
        raise ValueError(f"Expected {old_log_prob.shape[0]} uid values, got {len(uid_list)}.")

    dtype = old_log_prob.dtype
    device = old_log_prob.device
    response_mask = response_mask.to(dtype=dtype)
    token_level_rewards = token_level_rewards.to(dtype=dtype)
    ref_log_prob = ref_log_prob.to(dtype=dtype)
    old_token_kl = old_token_kl.to(dtype=dtype)
    beta_tensor = torch.as_tensor(beta, dtype=dtype, device=device)

    final_reward = _masked_sequence_sum(token_level_rewards, response_mask)
    old_kl_sum = _masked_sequence_sum(old_token_kl, response_mask)
    old_log_ratio_sum = _masked_sequence_sum(old_log_prob - ref_log_prob, response_mask)
    soft_return = final_reward - beta_tensor * old_kl_sum

    baseline = compute_grouped_baseline_by_uid(soft_return, uid_list)
    if cap_v0 is not None:
        baseline = baseline.clamp_max(torch.as_tensor(cap_v0, dtype=baseline.dtype, device=baseline.device))

    metrics = {
        "soft_v0": baseline.mean().detach(),
        "soft_final_reward": final_reward.mean().detach(),
        "soft_old_kl_sum": old_kl_sum.mean().detach(),
        "soft_old_log_ratio_sum": old_log_ratio_sum.mean().detach(),
    }
    return baseline.detach(), metrics


def compute_soft_trajectory_vimpo_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    old_token_kl: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    soft_value_baseline: torch.Tensor,
    beta: float,
    loss_agg_mode: str,
    global_batch_info: dict,
    value_loss_type: str = "squared_error",
    huber_delta: float = 1.0,
    value_loss_normalized: bool = False,
    value_loss_coeff: float = 1.0,
    value_loss_norm_denom: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute trajectory-level soft-value VIMPO loss.

    The differentiable term is beta * sum_t log(pi_current/ref). The detached
    baseline is supplied by ``compute_soft_value_baseline_by_uid`` and should be
    computed from old-policy rollout quantities.
    """
    if log_prob.shape != ref_log_prob.shape:
        raise ValueError(f"{log_prob.shape=} does not match {ref_log_prob.shape=}")
    if log_prob.shape != old_token_kl.shape:
        raise ValueError(f"{log_prob.shape=} does not match {old_token_kl.shape=}")
    if log_prob.shape != token_level_rewards.shape:
        raise ValueError(f"{log_prob.shape=} does not match {token_level_rewards.shape=}")
    if log_prob.shape != response_mask.shape:
        raise ValueError(f"{log_prob.shape=} does not match {response_mask.shape=}")
    if soft_value_baseline.shape != log_prob.shape[:1]:
        raise ValueError(f"{soft_value_baseline.shape=} does not match batch shape {log_prob.shape[:1]}.")

    dtype = log_prob.dtype
    device = log_prob.device
    response_mask = response_mask.to(dtype=dtype)
    ref_log_prob = ref_log_prob.to(dtype=dtype)
    old_token_kl = old_token_kl.to(dtype=dtype)
    token_level_rewards = token_level_rewards.to(dtype=dtype)
    soft_value_baseline = soft_value_baseline.to(dtype=dtype, device=device).detach()
    beta_tensor = torch.as_tensor(beta, dtype=dtype, device=device)

    vimpo_kl_agg = agg_loss(
        loss_mat=old_token_kl,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **global_batch_info,
    )
    s_current = beta_tensor * _masked_sequence_sum(log_prob - ref_log_prob, response_mask)
    final_reward = _masked_sequence_sum(token_level_rewards, response_mask)
    target = final_reward - soft_value_baseline
    raw_loss, terminal_stats = _compute_terminal_value_loss_stats(
        s_current,
        target,
        value_loss_type=value_loss_type,
        huber_delta=huber_delta,
    )
    loss, value_loss_metrics = _scale_value_loss(
        raw_loss,
        value_loss_normalized=value_loss_normalized,
        value_loss_coeff=value_loss_coeff,
        value_loss_norm_denom=value_loss_norm_denom,
    )

    metrics = {
        "vimpo_kl": vimpo_kl_agg.detach(),
        "vimpo_terminal_mse": loss.detach(),
        "vimpo_value_loss_raw": value_loss_metrics["value_loss_raw"].detach(),
        "vimpo_value_loss_for_backward": value_loss_metrics["value_loss_for_backward"].detach(),
        "vimpo_value_loss_coeff": value_loss_metrics["value_loss_coeff"].detach(),
        "vimpo_value_loss_normalized": value_loss_metrics["value_loss_normalized"].detach(),
        "vimpo_soft_v0": soft_value_baseline.mean().detach(),
        "vimpo_soft_s_current": s_current.mean().detach(),
        "vimpo_soft_target": target.mean().detach(),
        "vimpo_soft_final_reward": final_reward.mean().detach(),
        "terminal_rmse_over_target_rms": terminal_stats["terminal_rmse_over_target_rms"].detach(),
        "vimpo_value_loss_type_huber": terminal_stats["value_loss_type_huber"].detach(),
        "vimpo_value_huber_delta": terminal_stats["value_huber_delta"].detach(),
        "vimpo_terminal_squared_error_mean": terminal_stats["terminal_squared_error_mean"].detach(),
        "vimpo_terminal_huber_loss_mean": terminal_stats["terminal_huber_loss_mean"].detach(),
        "vimpo_terminal_huber_clipped_frac": terminal_stats["terminal_huber_clipped_frac"].detach(),
        "vimpo_terminal_abs_error_mean": terminal_stats["terminal_abs_error_mean"].detach(),
    }
    if "value_loss_norm_denom" in value_loss_metrics:
        metrics["vimpo_value_loss_norm_denom"] = value_loss_metrics["value_loss_norm_denom"].detach()
    return loss, vimpo_kl_agg, metrics


def compose_vimpo_policy_loss(
    pg_loss: torch.Tensor | None,
    vimpo_terminal_mse: torch.Tensor | None,
    use_grpo_actor: bool,
    use_ppo_actor: bool,
    vimpo_actor_coeff: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine VIMPO with optional PPO/GRPO policy loss."""
    policy_loss = pg_loss if pg_loss is not None else torch.zeros([], device=device)
    metrics: dict[str, float] = {}
    if vimpo_terminal_mse is None:
        return policy_loss, metrics
    if use_grpo_actor or use_ppo_actor:
        policy_loss = vimpo_terminal_mse + vimpo_actor_coeff * policy_loss
        metrics["vimpo_actor_coeff"] = vimpo_actor_coeff
    else:
        policy_loss = vimpo_terminal_mse
    return policy_loss, metrics


def compute_detached_vimpo_ppo_targets(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    gamma: float,
    lam: float,
    whiten_adv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build detached VIMPO PPO targets from the intrinsic token TD residual.

    The actor advantage uses the VIMPO token term directly. For the final-reward
    VIMPO objective with gamma=1, the external reward terms cancel from the TD
    residual, so the PPO actor branch does not need token-level rewards.
    """
    if gamma <= 0:
        raise ValueError(f"Expected gamma > 0, got {gamma}")
    if not (0 <= lam <= 1):
        raise ValueError(f"Expected lam in [0, 1], got {lam}")

    with torch.no_grad():
        vimpo_token_term = build_vimpo_token_term(
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
            token_kl=token_kl,
            response_mask=response_mask,
            beta=beta,
        )
        dtype = vimpo_token_term.dtype
        device = vimpo_token_term.device
        response_mask = response_mask.to(dtype=dtype)
        gamma_t = torch.as_tensor(gamma, dtype=dtype, device=device)
        lam_t = torch.as_tensor(lam, dtype=dtype, device=device)

        values = torch.zeros_like(vimpo_token_term)
        next_value = torch.zeros(vimpo_token_term.shape[0], dtype=dtype, device=device)

        for t in reversed(range(vimpo_token_term.shape[1])):
            candidate = gamma_t * next_value - vimpo_token_term[:, t]
            value_t = candidate * response_mask[:, t] + next_value * (1 - response_mask[:, t])
            values[:, t] = value_t
            next_value = value_t

        nextvalues = torch.zeros_like(next_value)
        lastgaelam = torch.zeros_like(next_value)
        advantages_reversed = []

        for t in reversed(range(vimpo_token_term.shape[1])):
            delta = gamma_t * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma_t * lam_t * lastgaelam
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
            advantages_reversed.append(lastgaelam)

        raw_advantages = torch.stack(advantages_reversed[::-1], dim=1) * response_mask
        returns = raw_advantages + values
        if whiten_adv:
            advantages = masked_whiten_distributed(raw_advantages, response_mask) * response_mask
        else:
            advantages = raw_advantages
        values = values * response_mask
        returns = returns * response_mask
    return advantages, values, returns, raw_advantages


def compute_sampled_token_kl_from_log_probs(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    estimator: str,
    detach_kl: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute sampled-token KL estimates from current/ref log-probs."""
    if log_prob.shape != ref_log_prob.shape:
        raise ValueError(f"{log_prob.shape=} does not match {ref_log_prob.shape=}")
    if output_dtype is None:
        output_dtype = log_prob.dtype

    def _compute() -> torch.Tensor:
        log_ratio = ref_log_prob - log_prob
        if estimator == "k1":
            token_kl = -log_ratio
        elif estimator == "k2":
            token_kl = 0.5 * log_ratio.square()
        elif estimator == "k3":
            token_kl = torch.exp(log_ratio) - 1 - log_ratio
        else:
            raise NotImplementedError(
                f"Sampled VIMPO KL estimator {estimator!r} is not supported. Expected one of 'k1', 'k2', 'k3'."
            )
        return token_kl.to(output_dtype)

    if detach_kl:
        with torch.no_grad():
            return _compute()
    return _compute()


def compute_exact_flat_kl_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    vocab_chunk_size: int = 0,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute exact KL(policy || ref) for flattened token logits."""
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"{policy_logits.shape=} does not match {ref_logits.shape=}")
    if policy_logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got {policy_logits.ndim=}")

    if output_dtype is None:
        output_dtype = compute_dtype if compute_dtype is not None else policy_logits.dtype

    if compute_dtype is not None:
        policy_logits = policy_logits.to(compute_dtype)
        ref_logits = ref_logits.to(compute_dtype)

    if vocab_chunk_size is None or vocab_chunk_size <= 0 or vocab_chunk_size >= policy_logits.shape[-1]:
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        return torch.sum(torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs), dim=-1).to(output_dtype)

    policy_lse = _chunked_logsumexp(policy_logits, vocab_chunk_size)
    ref_lse = _chunked_logsumexp(ref_logits, vocab_chunk_size)
    token_kl = torch.zeros(policy_logits.shape[0], device=policy_logits.device, dtype=torch.float32)

    vocab_size = policy_logits.shape[-1]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        policy_log_probs_chunk = policy_logits[:, start:end] - policy_lse
        ref_log_probs_chunk = ref_logits[:, start:end] - ref_lse
        policy_probs_chunk = torch.exp(policy_log_probs_chunk)
        token_kl = token_kl + torch.sum(
            policy_probs_chunk * (policy_log_probs_chunk - ref_log_probs_chunk),
            dim=-1,
            dtype=torch.float32,
        )

    return token_kl.to(output_dtype)


def compute_candidate_topk_flat_kl_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    candidate_kl_topk: int,
    vocab_chunk_size: int = 0,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute a top-k candidate approximation to KL(policy || ref).

    This keeps only the current-policy top-k tokens in the KL summation while using
    the full-vocabulary log normalizers for both policy and reference distributions.
    """
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"{policy_logits.shape=} does not match {ref_logits.shape=}")
    if policy_logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got {policy_logits.ndim=}")
    if candidate_kl_topk <= 0:
        raise ValueError(f"Expected candidate_kl_topk > 0, got {candidate_kl_topk}")

    if output_dtype is None:
        output_dtype = compute_dtype if compute_dtype is not None else policy_logits.dtype

    if compute_dtype is not None:
        policy_logits = policy_logits.to(compute_dtype)
        ref_logits = ref_logits.to(compute_dtype)

    vocab_size = policy_logits.shape[-1]
    topk = min(candidate_kl_topk, vocab_size)

    if vocab_chunk_size is None or vocab_chunk_size <= 0 or vocab_chunk_size >= vocab_size:
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        topk_indices = torch.topk(policy_logits, k=topk, dim=-1, largest=True, sorted=False).indices
        policy_topk_log_probs = torch.gather(policy_log_probs, dim=-1, index=topk_indices)
        ref_topk_log_probs = torch.gather(ref_log_probs, dim=-1, index=topk_indices)
        return torch.sum(
            torch.exp(policy_topk_log_probs) * (policy_topk_log_probs - ref_topk_log_probs),
            dim=-1,
        ).to(output_dtype)

    policy_lse = _chunked_logsumexp(policy_logits, vocab_chunk_size)
    ref_lse = _chunked_logsumexp(ref_logits, vocab_chunk_size)
    topk_indices = torch.topk(policy_logits, k=topk, dim=-1, largest=True, sorted=False).indices
    policy_topk_logits = torch.gather(policy_logits, dim=-1, index=topk_indices)
    ref_topk_logits = torch.gather(ref_logits, dim=-1, index=topk_indices)
    policy_topk_log_probs = policy_topk_logits - policy_lse
    ref_topk_log_probs = ref_topk_logits - ref_lse
    return torch.sum(
        torch.exp(policy_topk_log_probs) * (policy_topk_log_probs - ref_topk_log_probs),
        dim=-1,
        dtype=torch.float32,
    ).to(output_dtype)


def compute_vimpo_flat_kl_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    estimator: str,
    detach_kl: bool,
    vocab_chunk_size: int = 0,
    candidate_kl_topk: int = 0,
    candidate_kl_M: int = 0,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute VIMPO token KL from flattened logits using the configured estimator."""

    def _compute() -> torch.Tensor:
        if estimator == "exact":
            return compute_exact_flat_kl_from_logits(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                vocab_chunk_size=vocab_chunk_size,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
        if estimator == "candidate_set":
            if candidate_kl_M != 0:
                raise NotImplementedError(
                    "VIMPO candidate_set KL currently supports policy top-k only; candidate_kl_M must be 0."
                )
            return compute_candidate_topk_flat_kl_from_logits(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                candidate_kl_topk=candidate_kl_topk,
                vocab_chunk_size=vocab_chunk_size,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
        raise NotImplementedError(
            f"VIMPO KL estimator {estimator!r} is not supported yet. Only 'exact' and 'candidate_set' are implemented."
        )

    if detach_kl:
        with torch.no_grad():
            return _compute()
    return _compute()


def compute_exact_token_kl_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_chunk_size: int = 0,
) -> torch.Tensor:
    """Compute exact tokenwise KL(policy || ref) over the full vocabulary.

    Args:
        policy_logits: Current-policy logits with shape [batch, response_len, vocab].
        ref_logits: Reference-policy logits with shape [batch, response_len, vocab].
        response_mask: Response mask with shape [batch, response_len].
        vocab_chunk_size: Optional chunk size over the vocabulary dimension. Values <= 0 disable chunking.

    Returns:
        Tensor with shape [batch, response_len]. Masked positions are zeroed.
    """
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"{policy_logits.shape=} does not match {ref_logits.shape=}")
    if policy_logits.ndim != 3:
        raise ValueError(f"Expected 3D logits, got {policy_logits.ndim=}")

    response_mask = response_mask.to(dtype=policy_logits.dtype)
    token_kl = compute_exact_flat_kl_from_logits(
        policy_logits.reshape(-1, policy_logits.shape[-1]),
        ref_logits.reshape(-1, ref_logits.shape[-1]),
        vocab_chunk_size=vocab_chunk_size,
    ).reshape_as(policy_logits[..., 0])
    return token_kl * response_mask
