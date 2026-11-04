# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from recipe.vimpo.vimpo_core_algos import (
    build_vimpo_token_term,
    compute_detached_vimpo_ppo_targets,
    compute_reward_baseline_by_uid,
    compute_soft_value_baseline_by_uid,
    compute_soft_trajectory_vimpo_loss,
    compute_vimpo_adaptive_actor_coeff,
    compute_vimpo_terminal_mse_loss,
)


def test_vimpo_terminal_squared_error_matches_existing_mse():
    prediction = torch.tensor([[0.2], [3.0], [-2.0]])
    target = torch.zeros_like(prediction)
    response_mask = torch.ones_like(prediction)

    loss, metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=prediction,
        token_level_rewards=target,
        response_mask=response_mask,
        gamma=1.0,
        reward_baseline=0.0,
        value_loss_type="squared_error",
        huber_delta=0.5,
    )

    expected_loss = 0.5 * prediction.squeeze(-1).square().mean()
    expected_mse = prediction.squeeze(-1).square().mean()
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(metrics["terminal_squared_error_mean"], expected_mse)
    assert torch.allclose(metrics["terminal_half_squared_error_mean"], expected_loss)
    assert metrics["value_loss_type_huber"].item() == 0.0


def test_vimpo_terminal_huber_matches_piecewise_formula_and_reports_clipping():
    prediction = torch.tensor([[0.2], [3.0], [-2.0]])
    target = torch.zeros_like(prediction)
    response_mask = torch.ones_like(prediction)
    delta = 0.5

    loss, metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=prediction,
        token_level_rewards=target,
        response_mask=response_mask,
        gamma=1.0,
        reward_baseline=0.0,
        value_loss_type="huber",
        huber_delta=delta,
    )

    residual = prediction.squeeze(-1)
    abs_residual = residual.abs()
    expected_per_seq = torch.where(
        abs_residual <= delta,
        0.5 * residual.square(),
        delta * (abs_residual - 0.5 * delta),
    )
    assert torch.allclose(loss, expected_per_seq.mean())
    assert torch.allclose(metrics["terminal_huber_loss_mean"], expected_per_seq.mean())
    assert torch.allclose(metrics["terminal_huber_clipped_frac"], torch.tensor(2.0 / 3.0))
    assert torch.allclose(metrics["terminal_abs_error_mean"], abs_residual.mean())
    assert metrics["value_loss_type_huber"].item() == 1.0
    assert metrics["value_huber_delta"].item() == delta


def test_vimpo_terminal_huber_delta_must_be_positive():
    tensor = torch.ones(1, 1)

    with pytest.raises(ValueError, match="vimpo_value_huber_delta"):
        compute_vimpo_terminal_mse_loss(
            vimpo_token_term=tensor,
            token_level_rewards=torch.zeros_like(tensor),
            response_mask=torch.ones_like(tensor),
            gamma=1.0,
            value_loss_type="huber",
            huber_delta=0.0,
        )


def test_vimpo_terminal_value_loss_type_must_be_known():
    tensor = torch.ones(1, 1)

    with pytest.raises(ValueError, match="vimpo_value_loss_type"):
        compute_vimpo_terminal_mse_loss(
            vimpo_token_term=tensor,
            token_level_rewards=torch.zeros_like(tensor),
            response_mask=torch.ones_like(tensor),
            gamma=1.0,
            value_loss_type="l1",
            huber_delta=1.0,
        )


def test_vimpo_terminal_value_loss_normalization_scales_squared_error():
    prediction = torch.tensor([[3.0], [0.0]])
    rewards = torch.tensor([[1.0], [3.0]])
    response_mask = torch.ones_like(prediction)

    loss, metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=prediction,
        token_level_rewards=rewards,
        response_mask=response_mask,
        gamma=1.0,
        reward_baseline=0.0,
        value_loss_type="squared_error",
        huber_delta=1.0,
        value_loss_normalized=True,
        value_loss_coeff=2.0,
        value_loss_norm_denom=5.0,
    )

    raw_loss = torch.tensor(3.25)
    assert torch.allclose(metrics["value_loss_raw"], raw_loss)
    assert torch.allclose(metrics["value_loss_norm_denom"], torch.tensor(5.0))
    assert torch.allclose(metrics["value_loss_for_backward"], torch.tensor(1.3))
    assert torch.allclose(loss, torch.tensor(1.3))


def test_vimpo_terminal_value_loss_normalization_scales_huber():
    prediction = torch.tensor([[3.0], [0.0]])
    rewards = torch.tensor([[1.0], [3.0]])
    response_mask = torch.ones_like(prediction)

    loss, metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=prediction,
        token_level_rewards=rewards,
        response_mask=response_mask,
        gamma=1.0,
        reward_baseline=0.0,
        value_loss_type="huber",
        huber_delta=1.0,
        value_loss_normalized=True,
        value_loss_coeff=0.5,
        value_loss_norm_denom=5.0,
    )

    raw_loss = torch.tensor(2.0)
    assert torch.allclose(metrics["value_loss_raw"], raw_loss)
    assert torch.allclose(metrics["value_loss_for_backward"], torch.tensor(0.2))
    assert torch.allclose(loss, torch.tensor(0.2))


def test_vimpo_terminal_value_loss_normalization_rejects_tiny_denom():
    tensor = torch.ones(1, 1)

    with pytest.raises(ValueError, match="target MSE denominator >= 1e-4"):
        compute_vimpo_terminal_mse_loss(
            vimpo_token_term=tensor,
            token_level_rewards=torch.zeros_like(tensor),
            response_mask=torch.ones_like(tensor),
            gamma=1.0,
            value_loss_normalized=True,
            value_loss_norm_denom=1e-5,
        )


def test_vimpo_terminal_value_loss_coeff_can_zero_value_loss():
    tensor = torch.ones(1, 1)

    loss, metrics = compute_vimpo_terminal_mse_loss(
        vimpo_token_term=tensor,
        token_level_rewards=torch.zeros_like(tensor),
        response_mask=torch.ones_like(tensor),
        gamma=1.0,
        value_loss_coeff=0.0,
    )

    assert torch.allclose(metrics["value_loss_raw"], torch.tensor(0.5))
    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(metrics["value_loss_for_backward"], torch.tensor(0.0))


def test_soft_vimpo_uses_configured_value_loss_type():
    log_prob = torch.tensor([[1.0]])
    ref_log_prob = torch.zeros_like(log_prob)
    old_token_kl = torch.zeros_like(log_prob)
    token_level_rewards = torch.zeros_like(log_prob)
    response_mask = torch.ones_like(log_prob)
    soft_value_baseline = torch.zeros(1)

    loss, _, metrics = compute_soft_trajectory_vimpo_loss(
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        old_token_kl=old_token_kl,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        soft_value_baseline=soft_value_baseline,
        beta=2.0,
        loss_agg_mode="token-mean",
        global_batch_info={},
        value_loss_type="huber",
        huber_delta=0.5,
    )

    assert torch.allclose(loss, torch.tensor(0.875))
    assert torch.allclose(metrics["vimpo_terminal_huber_loss_mean"], torch.tensor(0.875))
    assert metrics["vimpo_value_loss_type_huber"].item() == 1.0


def test_vimpo_cap_v0_caps_raw_and_soft_prompt_baselines():
    token_level_rewards = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    response_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uids = ["prompt-a", "prompt-a"]

    raw_baseline = compute_reward_baseline_by_uid(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        uids=uids,
        cap_v0=0.5,
    )

    assert torch.allclose(raw_baseline, torch.tensor([0.5, 0.5]))

    soft_baseline, metrics = compute_soft_value_baseline_by_uid(
        old_log_prob=torch.zeros_like(token_level_rewards),
        ref_log_prob=torch.zeros_like(token_level_rewards),
        old_token_kl=torch.zeros_like(token_level_rewards),
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        beta=0.1,
        uids=uids,
        cap_v0=0.25,
    )

    assert torch.allclose(soft_baseline, torch.tensor([0.25, 0.25]))
    assert torch.allclose(metrics["soft_v0"], torch.tensor(0.25))


def test_vimpo_ppo_actor_advantage_does_not_need_rewards_when_gamma_one():
    log_prob = torch.tensor([[0.1, -0.2, 0.3]])
    ref_log_prob = torch.tensor([[0.0, -0.1, 0.1]])
    token_kl = torch.tensor([[0.02, 0.03, 0.04]])
    response_mask = torch.ones_like(log_prob)
    token_level_rewards = torch.tensor([[0.0, 0.0, 1.0]])
    beta = 0.5
    lam = 0.95

    _, _, _, raw_advantages = compute_detached_vimpo_ppo_targets(
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        token_kl=token_kl,
        response_mask=response_mask,
        beta=beta,
        gamma=1.0,
        lam=lam,
        whiten_adv=False,
    )

    vimpo_token_term = build_vimpo_token_term(
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        token_kl=token_kl,
        response_mask=response_mask,
        beta=beta,
    )
    values = torch.zeros_like(vimpo_token_term)
    next_value = torch.zeros(vimpo_token_term.shape[0])
    for t in reversed(range(vimpo_token_term.shape[1])):
        values[:, t] = token_level_rewards[:, t] + next_value - vimpo_token_term[:, t]
        next_value = values[:, t]

    expected_reversed = []
    next_value = torch.zeros(vimpo_token_term.shape[0])
    lastgaelam = torch.zeros(vimpo_token_term.shape[0])
    for t in reversed(range(vimpo_token_term.shape[1])):
        delta = token_level_rewards[:, t] + next_value - values[:, t]
        lastgaelam = delta + lam * lastgaelam
        next_value = values[:, t]
        expected_reversed.append(lastgaelam)
    expected = torch.stack(expected_reversed[::-1], dim=1)

    assert torch.allclose(raw_advantages, expected)


def test_adaptive_actor_coeff_uses_zero_start_ema_and_detached_values():
    value_rms = torch.tensor(2.0, requires_grad=True)
    actor_rms = torch.tensor(10.0, requires_grad=True)

    coeff, value_ema, actor_ema = compute_vimpo_adaptive_actor_coeff(
        value_logit_grad_rms=value_rms,
        actor_logit_grad_rms_unscaled=actor_rms,
        value_logit_grad_rms_ema=0.0,
        actor_logit_grad_rms_unscaled_ema=0.0,
        alpha=0.15,
        ema_decay=0.9,
        eps=1e-12,
    )

    assert isinstance(coeff, float)
    assert value_ema == pytest.approx(0.2)
    assert actor_ema == pytest.approx(1.0)
    assert coeff == pytest.approx(0.03)


def test_adaptive_actor_coeff_eps_handles_zero_actor_scale():
    coeff, value_ema, actor_ema = compute_vimpo_adaptive_actor_coeff(
        value_logit_grad_rms=2.0,
        actor_logit_grad_rms_unscaled=0.0,
        value_logit_grad_rms_ema=0.0,
        actor_logit_grad_rms_unscaled_ema=0.0,
        alpha=0.15,
        ema_decay=0.9,
        eps=1e-6,
    )

    assert value_ema == pytest.approx(0.2)
    assert actor_ema == pytest.approx(0.0)
    assert coeff == pytest.approx(30000.0)
