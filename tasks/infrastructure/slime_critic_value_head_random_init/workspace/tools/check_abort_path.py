#!/usr/bin/env python3
"""Offline check: an aborted episode must survive slime's train-data builder.

`_convert_samples_to_train_data` decides whether an optional per-sample field
is present by looking at `samples[0]` alone, and then indexes *every* sample in
the batch. So the shape that breaks is a batch mixing one normal episode with
one that gave up -- which is exactly what a rollout looks like whenever SGLang
aborts a straggler. The failure surfaces as a `KeyError` or a `TypeError` deep
inside slime, an hour into a run, and costs the whole rollout.

Reproducing that shape here costs seconds. Run it after touching any giving-up
path in `et/rollout.py`:

    python3 tools/check_abort_path.py    # inside the slime image

No GPU, no model and no network: the tokenizer is stubbed out, because the only
thing `_abort` wants from it is an end-of-text id.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import et.rollout as R  # noqa: E402
from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

EOS = 151645  # Qwen3's <|im_end|>; any valid id would do, it is never trained on


class _FakeTokenizer:
    eos_token_id = EOS


class _FakeState:
    """Stands in for slime's `GenerateState`, which would load a real tokenizer."""

    def __init__(self, args):
        self.tokenizer = _FakeTokenizer()


R.GenerateState = _FakeState

ARGS = Namespace(
    rollout_top_p=1.0,
    rollout_max_response_len=2048,
    partial_rollout=False,
    use_rollout_routing_replay=False,
)


def normal_sample(index: int) -> Sample:
    """An episode that ran a turn and was scored -- opens every slime gate."""
    sample = Sample(index=index, prompt="p", label="task_normal")
    sample.tokens = [1, 2, 3]
    sample.append_response_tokens(ARGS, tokens=[10, 11], log_probs=[-0.1, -0.2], trainable=True, text="a")
    sample.append_response_tokens(ARGS, tokens=[12], trainable=False, text="obs")
    sample.reward = 1.0
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {"round_number": 1, "et_task": "task_normal"}
    return sample


def aborted_sample(index: int, turns: int) -> Sample:
    """An episode that gave up -- before generating anything, or after a turn."""
    sample = Sample(index=index, prompt="p", label="task_aborted")
    sample.tokens = [1, 2, 3]
    if turns:
        sample.append_response_tokens(ARGS, tokens=[20], log_probs=[-0.3], trainable=True, text="b")
    return R._abort(ARGS, sample, turns, et_finish="sglang_abort")


def main() -> int:
    failures: list[str] = []

    for turns in (0, 1):
        sample = aborted_sample(1, turns)
        where = f"abort after {turns} turn(s)"
        if sample.reward is None:
            failures.append(f"{where}: reward is None (slime's eval metrics average over the whole split)")
        if not sample.metadata or "round_number" not in sample.metadata:
            failures.append(f"{where}: metadata['round_number'] missing (ray/rollout.py:382)")
        if sample.rollout_log_probs is None:
            failures.append(f"{where}: rollout_log_probs is None (ray/rollout.py:386)")
        elif len(sample.rollout_log_probs) != sample.response_length:
            failures.append(
                f"{where}: rollout_log_probs {len(sample.rollout_log_probs)} "
                f"!= response_length {sample.response_length}"
            )
        if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
            failures.append(f"{where}: loss_mask inconsistent with response_length")
        if turns == 0 and sum(sample.loss_mask or []):
            failures.append(f"{where}: an episode that generated nothing must contribute no trainable token")
        print(
            f"  {where}: response_length={sample.response_length} loss_mask={sample.loss_mask} "
            f"log_probs={sample.rollout_log_probs} round_number={sample.metadata.get('round_number')}"
        )

    # The batch slime actually builds: a normal sample first, so every optional
    # field's gate opens on `samples[0]`, then the ones that gave up.
    samples = [normal_sample(0), aborted_sample(1, 0), aborted_sample(2, 1)]
    for sample in samples:
        if sample.loss_mask is None:  # slime does this itself, before the gates
            sample.loss_mask = [1] * sample.response_length

    train_data = {
        "tokens": [s.tokens for s in samples],
        "response_lengths": [s.response_length for s in samples],
        "rewards": [s.reward for s in samples],
        "loss_masks": [s.loss_mask for s in samples],
    }
    if samples[0].metadata and "round_number" in samples[0].metadata:
        try:
            train_data["round_number"] = [s.metadata["round_number"] for s in samples]
        except KeyError as exc:
            failures.append(f"round_number gate: KeyError {exc}")
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [s.rollout_log_probs for s in samples]
    # These three gates must stay shut: this rollout never populates them, and a
    # sample that opened one would take the batch down the same way.
    for field in ("rollout_routed_experts", "train_metadata", "teacher_log_probs"):
        if getattr(samples[0], field, None) is not None:
            failures.append(f"{field} gate unexpectedly open")

    try:
        tensorize_rollout_data_for_training(train_data)
        print("  tensorize: OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"tensorize: {type(exc).__name__}: {exc}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print("  -", failure)
        return 1
    print("\nPASS: a batch mixing normal and aborted episodes survives slime's train-data builder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
