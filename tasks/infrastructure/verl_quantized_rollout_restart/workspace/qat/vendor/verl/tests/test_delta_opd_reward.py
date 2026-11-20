import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import (
    _compute_delta_opd_metrics,
    _compute_ref_teacher_overlap_metrics,
    _compute_student_weighted_teacher_logprob_metrics,
    _pop_direct_opd_rollout_options,
)
from verl.utils.torch_functional import _logits_row_chunk, logprobs_for_ids_fp32, logprobs_from_logits_v2
from verl.workers.actor.dp_actor import (
    _compute_delta_opd_rm_scores,
    _positive_temperature as actor_positive_temperature,
)
from verl.workers.fsdp_workers import (
    RewardModelWorker,
    _compute_student_topk_in_teacher_top_p_mask,
    _positive_temperature as reward_positive_temperature,
)


def test_fp32_logprobs_strided_response_slice_is_bounded_and_matches_reference():
    """A strided [batch, response, vocab] view must not require full flattening."""
    full_logits = torch.randn(2, 9, 23, dtype=torch.bfloat16)
    logits = full_logits[:, 2:8, :]  # the same non-contiguous slice used by workers
    ids = torch.randint(0, logits.size(-1), logits.shape[:-1])

    output = logprobs_for_ids_fp32(logits, ids, chunk_size=2, temperature=0.73)
    reference = (logits.float() / 0.73).log_softmax(dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

    assert not logits.is_contiguous()
    assert output.dtype == torch.float32
    assert torch.allclose(output, reference, atol=2e-5, rtol=0)


def test_logit_row_chunk_selects_only_requested_strided_rows():
    full_logits = torch.arange(2 * 9 * 7, dtype=torch.float32).reshape(2, 9, 7)
    logits = full_logits[:, 2:8, :]

    chunk = _logits_row_chunk(logits, 3, 6, tuple(logits.shape[:-1]))
    # ``start``/``stop`` are logical row indices of the supplied view (whose
    # leading shape is [2, 6]), not indices into the original unsliced tensor.
    expected = torch.stack((logits[0, 3], logits[0, 4], logits[0, 5]))

    assert chunk.shape == (3, logits.size(-1))
    assert torch.equal(chunk, expected)


def test_global_logprobs_v2_keeps_legacy_bfloat16_dtype():
    """FP32 widening is scoped to the OPD helper, not standard PPO utility calls."""
    logits = torch.randn(2, 4, 11, dtype=torch.bfloat16)
    labels = torch.randint(0, logits.size(-1), (2, 4))

    output = logprobs_from_logits_v2(logits, labels)

    assert output.dtype == torch.bfloat16


def test_entropy_and_teacher_topk_accept_strided_logits():
    full_logits = torch.randn(2, 9, 17, dtype=torch.bfloat16)
    logits = full_logits[:, 2:8, :]
    worker = object.__new__(RewardModelWorker)

    entropy = worker._compute_entropy_safe(logits, chunk_size=2, temperature=0.8)
    reference_entropy = -(
        torch.softmax(logits.float() / 0.8, dim=-1)
        * torch.log_softmax(logits.float() / 0.8, dim=-1)
    ).sum(dim=-1)
    assert torch.allclose(entropy, reference_entropy, atol=2e-5, rtol=0)

    student_ids = torch.randint(0, logits.size(-1), (2, 6, 4), dtype=torch.int32)
    selected, counts, overlap, teacher_ids, teacher_logp, teacher_in_student = (
        worker._compute_teacher_top_k_log_probs(
            logits=logits,
            student_ids=student_ids,
            top_k=3,
            chunk_size=2,
            teacher_temperature=0.8,
        )
    )
    assert selected.shape == (12, 4)
    assert counts.shape == (12,)
    assert overlap.shape == (12, 4)
    assert teacher_ids.shape == teacher_logp.shape == teacher_in_student.shape == (12, 3)
    assert selected.dtype == teacher_logp.dtype == torch.float32
    assert teacher_ids.dtype == torch.long


def test_teacher_topk_validates_ids_chunks_and_handles_empty_rows():
    worker = object.__new__(RewardModelWorker)
    logits = torch.randn(0, 7, dtype=torch.bfloat16)
    empty_ids = torch.empty((0, 3), dtype=torch.int32)
    outputs = worker._compute_teacher_top_k_log_probs(
        logits=logits, student_ids=empty_ids, top_k=2, chunk_size=1
    )
    assert [tuple(value.shape) for value in outputs] == [
        (0, 3),
        (0,),
        (0, 3),
        (0, 2),
        (0, 2),
        (0, 2),
    ]

    nonempty_logits = torch.randn(2, 7, dtype=torch.bfloat16)
    ids = torch.zeros((2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="positive integer"):
        worker._compute_teacher_top_k_log_probs(nonempty_logits, ids, top_k=2, chunk_size=0)
    with pytest.raises(TypeError, match="int32/int64"):
        worker._compute_teacher_top_k_log_probs(nonempty_logits, ids.float(), top_k=2)
    with pytest.raises(ValueError, match="outside"):
        worker._compute_teacher_top_k_log_probs(
            nonempty_logits, torch.full_like(ids, 7), top_k=2
        )
    with pytest.raises(TypeError, match="int32/int64"):
        _compute_student_topk_in_teacher_top_p_mask(nonempty_logits, ids.float(), 0.9)


def test_fused_temperature_guard_rejects_nonpositive_or_nonfinite_values():
    for validator in (actor_positive_temperature, reward_positive_temperature):
        assert validator(0.75) == 0.75
        for invalid in (0.0, -1.0, float("nan"), float("inf"), True):
            with pytest.raises((TypeError, ValueError), match="positive finite"):
                validator(invalid)


def test_fp32_logprob_helper_rejects_boolean_chunk_size():
    logits = torch.randn(2, 5, dtype=torch.bfloat16)
    ids = torch.zeros(2, dtype=torch.long)
    with pytest.raises(TypeError, match="positive integer"):
        logprobs_for_ids_fp32(logits, ids, chunk_size=True)


def test_direct_opd_reward_uses_detached_student_topk_weights():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -1.0, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
    )

    expected_delta = teacher_rl_logp - teacher_ref_logp
    expected_weights = torch.softmax(student_logp.detach(), dim=-1)

    assert rm_scores.shape == torch.Size([1, 1, 3])
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert not rm_scores.requires_grad
    assert not delta.requires_grad


def test_direct_opd_reward_masks_invalid_candidates_before_weighting():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -torch.inf, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])
    valid_mask = torch.tensor([[[True, False, True]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
        valid_mask=valid_mask,
    )

    expected_delta = (teacher_rl_logp - teacher_ref_logp).masked_fill(~valid_mask, 0.0)
    masked_student_logp = student_logp.detach().masked_fill(~valid_mask, -torch.inf)
    expected_weights = torch.softmax(masked_student_logp, dim=-1)
    expected_weights = torch.nan_to_num(expected_weights, nan=0.0, posinf=0.0, neginf=0.0)

    assert torch.isfinite(rm_scores).all()
    assert torch.isfinite(delta).all()
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert rm_scores[0, 0, 1].item() == 0.0
    assert not rm_scores.requires_grad


def test_direct_opd_reward_rejects_mismatched_shapes():
    student_logp = torch.zeros(2, 3, 4)
    teacher_rl_logp = torch.zeros(2, 3, 4)
    teacher_ref_logp = torch.zeros(2, 3, 5)

    try:
        _compute_delta_opd_rm_scores(
            student_logp=student_logp,
            teacher_rl_logp=teacher_rl_logp,
            teacher_ref_logp=teacher_ref_logp,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched Direct-OPD shapes to raise ValueError")


def test_student_topk_teacher_top_p_mask_includes_threshold_crossing_token():
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    student_ids = torch.tensor([[0, 1, 2, 3]])

    mask = _compute_student_topk_in_teacher_top_p_mask(logits, student_ids, top_p=0.70)

    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.equal(mask, expected)


def test_teacher_top_p_intersec_strategy_returns_top_p_overlap_mask():
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    student_ids = torch.tensor([[0, 1, 2, 3]])

    (
        teacher_on_student_log_probs,
        valid_counts,
        overlap_mask,
        teacher_top_k_ids,
        teacher_top_k_log_probs,
        teacher_in_student_mask,
    ) = RewardModelWorker._compute_teacher_top_k_log_probs(
        None,
        logits=logits,
        student_ids=student_ids,
        top_k=2,
        strategy="top_p_intersec",
        top_p_intersec_p=0.70,
    )

    expected_log_probs = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    expected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    assert torch.allclose(teacher_on_student_log_probs, expected_log_probs)
    assert torch.equal(valid_counts, torch.tensor([2]))
    assert torch.equal(overlap_mask, expected_mask)
    assert torch.equal(teacher_top_k_ids, torch.tensor([[0, 1]]))
    assert torch.allclose(teacher_top_k_log_probs, torch.log(torch.tensor([[0.50, 0.30]])))
    assert torch.equal(teacher_in_student_mask, torch.tensor([[1.0, 1.0]]))


def test_direct_opd_rollout_options_are_removed_before_rollout_config_instantiation():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "reward_mode": "delta_opd",
                    "top_k_strategy": "top_p_intersec",
                    "top_p_intersec_p": 0.85,
                }
            }
        }
    )

    options = _pop_direct_opd_rollout_options(config)

    assert options == {"reward_mode": "delta_opd"}
    assert "reward_mode" not in config.actor_rollout_ref.rollout
    assert config.actor_rollout_ref.rollout.top_k_strategy == "top_p_intersec"
    assert config.actor_rollout_ref.rollout.top_p_intersec_p == 0.85


def test_direct_opd_metrics_include_weighted_pos_frac_and_teacher_gap():
    student_topk_log_probs = torch.log(torch.tensor([[[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]]))
    teacher_ref_on_student = torch.log(torch.tensor([[[0.3, 0.3, 0.4], [0.1, 0.7, 0.2]]]))
    response_mask = torch.tensor([[1, 1]])
    delta = teacher_on_student - teacher_ref_on_student
    rm_scores = torch.softmax(student_topk_log_probs, dim=-1) * delta

    metrics = {}
    metrics.update(
        _compute_delta_opd_metrics(
            delta=delta,
            rm_scores=rm_scores,
            response_mask=response_mask,
            student_topk_log_probs=student_topk_log_probs,
        )
    )
    metrics.update(
        _compute_student_weighted_teacher_logprob_metrics(
            student_topk_log_probs=student_topk_log_probs,
            teacher_on_student_log_probs=teacher_on_student,
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            response_mask=response_mask,
        )
    )
    metrics.update(
        _compute_ref_teacher_overlap_metrics(
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            teacher_ref_overlap_mask=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            response_mask=response_mask,
        )
    )

    assert "delta_opd/weighted_reward_mean" in metrics
    assert "delta_opd/topk_log_ratio_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_teacher_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_ref_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_logprob_gap" in metrics
    assert "delta_opd/ref_teacher_overlap_ratio" in metrics
