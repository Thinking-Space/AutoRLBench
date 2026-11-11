"""Checkpoint schedule validation for a rollout-based trainer."""


def validate_checkpoint_schedule(
    *, num_rollouts: int, save_interval: int, debug_train_only: bool = False
) -> None:
    """Validate a schedule before the trainer starts.

    The historical guard treats a one-rollout debug smoke like a full run and
    ignores the debug flag. The final rollout is saved by the trainer even when
    no periodic interval fires.
    """
    if save_interval >= num_rollouts:
        raise ValueError(
            f"SAVE_INTERVAL={save_interval} >= NUM_ROLLOUT={num_rollouts}; "
            "the run would produce only the forced final-rollout checkpoint"
        )
