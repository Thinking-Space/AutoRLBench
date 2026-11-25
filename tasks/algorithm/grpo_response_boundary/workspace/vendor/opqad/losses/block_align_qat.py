"""Per-block alignment between a model's quantized and full-precision selves.

The design that makes this cheap
--------------------------------
The obvious way to align a quantized student to a full-precision teacher is to
run the frozen teacher in the ``RewardModelWorker`` and ship its hidden states
to the actor.  For this profile that is 28 layers x [4, 18432, 3584] ~= 14.8 GB
per micro-batch across Ray, which is not viable.

For *quantization* QAT the reference does not have to be a separate model at
all.  The right target is "your quantized activations should match what you
yourself would have produced in full precision" -- so the reference forward is
the same weights with the fake quantizer switched off
(``TurboQuantAttnState.disabled()``).  That means:

  * no second 7B copy on the GPU,
  * no cross-worker transport,
  * the target tracks the weights as they train, which is what you want for
    "be robust to quantization" (as opposed to "imitate a fixed teacher").

Cost is one extra forward with no backward, provided the caller hands over the
quantized forward it already ran (``student_hidden``).  ``update_policy`` does;
without it this helper runs its own quantized forward, which is then backwarded
through -- an extra forward *and* backward rather than the intended one
forward.

This objective is also a *direct* differentiable loss, so it sidesteps the
detached-reward REINFORCE estimator whose missing term drives the entropy
collapse documented in ``block_align``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from opqad.losses.block_align import block_alignment_loss, select_layers, subsample_positions

__all__ = [
    "BlockAlignConfig",
    "compute_quantization_block_alignment",
    "reference_hidden_states",
]


class BlockAlignConfig:
    """Knobs for the alignment term.

    Args:
        layer_every: align every N-th block (plus the last).  All 28 blocks at
            the 7B profile cost ~14.8 GB of student activations that must live
            until backward; ``layer_every=4`` keeps 8 blocks (~4.2 GB).
        keep_fraction: fraction of response positions to score.  The target is
            a per-position expectation, so a random subset is unbiased.
        metric: ``"cosine"`` (scale-free), ``"nmse"``, ``"oprd_vanilla"``, ``"oprd_halfnorm"``, ``"token_huber_layer_norm"``, or ``"last_nmse"``.
            ``last_nmse`` is the nmse distance taken on the *last block only*
            (the pre-logits representation), not on every ``layer_every``-th
            block -- the "fit the final hidden state only" variant.
        coef: weight of the alignment term in the total loss.
        last_only: when True, align only the final layer's hidden state
            (an alternative to per-layer alignment).  Takes precedence over
            ``layer_every``/``metric``.
    """

    def __init__(
        self,
        layer_every: int = 4,
        keep_fraction: float = 0.125,
        metric: str = "cosine",
        coef: float = 1.0,
        last_only: bool = False,
        huber_delta: float = 1.0,
        layer_scales: Optional[Sequence[float]] = None,
    ):
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        self.layer_every = layer_every
        self.keep_fraction = keep_fraction
        self.metric = metric
        self.coef = coef
        self.last_only = last_only
        self.huber_delta = huber_delta
        self.layer_scales = layer_scales

    def metadata(self) -> dict:
        return {
            "block_align/layer_every": self.layer_every,
            "block_align/keep_fraction": self.keep_fraction,
            "block_align/metric": self.metric,
            "block_align/coef": self.coef,
            "block_align/last_only": self.last_only,
            "block_align/huber_delta": self.huber_delta,
            "block_align/fixed_layer_scales": float(self.layer_scales is not None),
        }


def _hidden_states(outputs) -> Sequence[torch.Tensor]:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise ValueError(
            "the model did not return hidden states; pass output_hidden_states=True"
        )
    # index 0 is the embedding output, 1..L are the block outputs
    return hs[1:]


@torch.no_grad()
def reference_hidden_states(
    model,
    state,
    model_inputs: Dict[str, Any],
    autocast_dtype: Optional[torch.dtype] = None,
    device_type: str = "cuda",
    frozen_model=None,
) -> Sequence[torch.Tensor]:
    """Full-precision block outputs to align the quantized forward against.

    ``frozen_model`` is the fix for a degeneracy in the same-weights version.
    With the reference recomputed from the *current* weights, both branches move
    together: under a straight-through estimator ``d f_q/d theta ~= d f/d theta``,
    so ``d (f_q - f)/d theta ~= 0`` and the distance is first-order invariant --
    the target runs away exactly as fast as the student approaches it.  Measured
    on a real run: the raw alignment loss stays in [0.00053, 0.00065] across a
    10x learning-rate range, a 10x coefficient range and 104 steps.

    Passing a frozen copy of the initial weights makes the target stationary, so
    the gradient has something to close.  ``frozen_model`` must not carry the
    quantizer hook -- it is full precision by construction, which is why
    ``state.disabled()`` is not needed for it.

    ``autocast_dtype`` must match whatever the student forward ran under.  The
    student and the reference differing only in autocast would put a floor
    under the loss that has nothing to do with quantization -- autocast keeps
    softmax and layer-norm in fp32, so a bf16-params-no-autocast forward is
    genuinely a different computation.
    """
    if frozen_model is not None:
        with torch.no_grad(), contextlib.ExitStack() as stack:
            if autocast_dtype is not None:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=autocast_dtype)
                )
            out = frozen_model(**model_inputs, output_hidden_states=True, use_cache=False)
        return [h.detach() for h in _hidden_states(out)]

    was_training = model.training
    model.eval()
    try:
        with contextlib.ExitStack() as stack:
            if autocast_dtype is not None:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=autocast_dtype)
                )
            stack.enter_context(state.disabled())
            out = model(**model_inputs, output_hidden_states=True, use_cache=False)
    finally:
        if was_training:
            model.train()
    return [h.detach() for h in _hidden_states(out)]


def compute_quantization_block_alignment(
    model,
    state,
    model_inputs: Dict[str, Any],
    response_mask: torch.Tensor,
    cfg: BlockAlignConfig,
    *,
    student_hidden: Optional[Sequence[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    device_type: str = "cuda",
    frozen_model=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Alignment loss between the quantized and full-precision forwards.

    Args:
        model: the actor.
        state: the installed ``TurboQuantAttnState``.
        model_inputs: kwargs for ``model(**model_inputs)``.
        response_mask: ``[B, T]`` over the positions to score.
        cfg: :class:`BlockAlignConfig`.
        student_hidden: block outputs from the quantized forward, if the caller
            already has them (``update_policy`` does).  When omitted a
            quantized forward is run here -- which costs an extra backward too,
            so pass them.
        autocast_dtype: the autocast dtype the student forward ran under, so
            the reference forward is the same computation.

    Returns:
        ``(loss, metrics)``.  ``loss`` is already scaled by ``cfg.coef``.
    """
    if not state.enabled:
        raise RuntimeError(
            "the quantizer is disabled; the alignment target would equal the "
            "student and the loss would be identically zero"
        )

    if student_hidden is None:
        with contextlib.ExitStack() as stack:
            if autocast_dtype is not None:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=autocast_dtype)
                )
            out = model(**model_inputs, output_hidden_states=True, use_cache=False)
        student_hidden = _hidden_states(out)

    teacher_hidden = reference_hidden_states(
        model, state, model_inputs, autocast_dtype=autocast_dtype,
        device_type=device_type, frozen_model=frozen_model,
    )
    if len(teacher_hidden) != len(student_hidden):
        raise ValueError(
            f"block count mismatch: student {len(student_hidden)}, reference {len(teacher_hidden)}"
        )

    if cfg.last_only:
        # Fit only the final block's hidden state (the pre-logits
        # representation).  This is the "last-nmse" variant: it gates the whole
        # quantization error budget on the one representation that directly
        # determines the logits, instead of re-scoring every layer.
        idx = [len(student_hidden) - 1]
    else:
        idx = select_layers(len(student_hidden), every=cfg.layer_every)
    mask = subsample_positions(response_mask, cfg.keep_fraction, generator=generator)

    loss, metrics = block_alignment_loss(
        [student_hidden[i] for i in idx],
        [teacher_hidden[i] for i in idx],
        mask,
        metric=cfg.metric,
        huber_delta=cfg.huber_delta,
        layer_scales=([cfg.layer_scales[i] for i in idx] if cfg.layer_scales is not None else None),
    )
    metrics["block_align/selected_layers"] = float(len(idx))
    metrics.update(cfg.metadata())
    return loss * cfg.coef, metrics
