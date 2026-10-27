import torch

from verl.trainer.ppo.core_algos import compute_opd_loss


def test_opd_loss_uses_gated_teacher_gap_on_selected_steps():
    log_prob = torch.tensor(
        [[-2.0, -1.0], [-0.5, -0.25]],
        requires_grad=True,
    )
    teacher_log_prob = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    response_mask = torch.ones_like(log_prob)
    step_mask = torch.tensor([1.0, 0.0])

    loss, active_ratio, gate_mean, gate_active_ratio, gap_mean = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        opd_step_mask=step_mask,
        gate_beta=1.0,
        loss_agg_mode="token-mean",
    )

    active_gap = teacher_log_prob[0] - log_prob.detach()[0]
    expected_gate = torch.sigmoid(active_gap)
    expected_loss = (expected_gate * (teacher_log_prob[0] - log_prob[0])).mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(gate_mean, expected_gate.mean())
    torch.testing.assert_close(gate_active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(gap_mean, torch.tensor(0.0))

    loss.backward()
    expected_grad = torch.zeros_like(log_prob)
    expected_grad[0] = -expected_gate / expected_gate.numel()
    torch.testing.assert_close(log_prob.grad, expected_grad)


def test_opd_loss_returns_zero_when_no_steps_are_selected():
    log_prob = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    teacher_log_prob = torch.tensor([[-1.0, -2.0]])
    response_mask = torch.ones_like(log_prob)
    step_mask = torch.tensor([0.0])

    outputs = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        opd_step_mask=step_mask,
    )

    for value in outputs:
        torch.testing.assert_close(value, torch.tensor(0.0))
