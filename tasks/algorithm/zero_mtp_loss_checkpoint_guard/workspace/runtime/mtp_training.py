"""Minimal policy plus auxiliary MTP loss path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Batch:
    policy_loss: float
    auxiliary_raw_loss: float


def compute_auxiliary_loss(batch: Batch, scaling_factor: float) -> float | None:
    """Return None when auxiliary MTP loss is disabled for the RL batch."""
    if scaling_factor == 0.0:
        return None
    return batch.auxiliary_raw_loss * scaling_factor


def train_step(batch: Batch, *, mtp_loss_scaling_factor: float) -> dict[str, float | int | None]:
    """Run one optimizer step.

    The defective path assumes the optional auxiliary loss is always present.
    """
    auxiliary_loss = compute_auxiliary_loss(batch, mtp_loss_scaling_factor)
    total_loss = batch.policy_loss + auxiliary_loss  # type: ignore[operator]
    return {
        "policy_loss": batch.policy_loss,
        "auxiliary_loss": auxiliary_loss,
        "total_loss": total_loss,
        "optimizer_steps": 1,
    }
