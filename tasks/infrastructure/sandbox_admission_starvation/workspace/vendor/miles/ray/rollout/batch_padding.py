"""Zero-loss batch-alignment padding for fan-out (multi-segment) rollouts.

Black-box fan-out rollouts (examples/black_box_rl*) split one trajectory into
K token segments sharing ``metadata["parent_traj_id"]``, so the flattened
sample count per rollout is a variable Σ Kᵢ that rarely divides
``global_batch_size``. The trim in ``postprocess_rollout_data`` would cut the
flat list from the tail — and the anchor segment (the one carrying the
trajectory's terminal reward) is each trajectory's LAST segment, exactly where
the trim lands. A trajectory severed from its anchor silently votes 0 into its
GRPO group.

Instead of trimming real data, ``postprocess_rollout_data`` pads fan-out
batches with minimal zero-loss samples. The padding target is dp_size (≤ dp-1
pads) when dynamic global batch size + token-packing micro-batching consume
the whole rollout in one optimizer step, else ``global_batch_size``:

* ``loss_mask=[0]`` and ``remove_sample=True`` — the padding contributes
  exactly 0 to the loss (``convert_samples_to_train_data`` re-zeroes the mask
  as a second line of defense);
* ``metadata["is_batch_pad"]=True`` — rollout metrics (``metrics.py``) and any
  fan-out-aware reward post-process hook skip it, so it never appears as a
  phantom zero-reward trajectory in zero_std / pass@k / group normalization;
* ``index`` is negative — distinguishable from real sample indices in dumps.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from miles.utils.replay_base import pad_value
from miles.utils.types import Sample


def is_batch_pad(sample: Sample) -> bool:
    return bool((sample.metadata or {}).get("is_batch_pad"))


def make_batch_pad_sample(template: Sample, pad_index: int) -> Sample:
    """Create a minimal zero-loss sample used only for batch alignment.

    Copied from a real sample of the same batch so that optional per-batch
    fields (rollout_log_probs, rollout_routed_experts, ...) mirror the
    template's presence — ``convert_samples_to_train_data`` keys those columns
    off ``samples[0]`` and requires every row to be structurally consistent.
    """
    sample = copy.copy(template)
    token_id = template.tokens[0] if template.tokens else 0
    sample.group_index = None
    # Negative indices keep padding distinguishable from real sample indices.
    sample.index = -(pad_index + 1)
    sample.prompt = ""
    sample.tokens = [token_id, token_id]
    sample.multimodal_inputs = None
    sample.multimodal_train_inputs = None
    sample.response = ""
    sample.response_length = 1
    sample.label = None
    sample.reward = 0.0
    sample.loss_mask = [0]
    sample.weight_versions = []
    sample.rollout_log_probs = [0.0] if template.rollout_log_probs is not None else None
    sample.rollout_routed_experts = None
    if template.rollout_routed_experts is not None:
        sample.rollout_routed_experts = np.full(
            (1, *template.rollout_routed_experts.shape[1:]),
            pad_value(template.rollout_routed_experts.dtype),
            dtype=template.rollout_routed_experts.dtype,
        )
    # Per-token side channels must not inherit the template's full-length
    # arrays (validate() checks them against response_length).
    sample.rollout_indexer_topk = None
    sample.teacher_log_probs = None
    sample.opd_reverse_kl = None
    # Padding must not inherit a real sample's engine affinity.
    sample.routing_key = None
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {"is_batch_pad": True, "raw_reward": 0.0}
    sample.train_metadata = copy.deepcopy(template.train_metadata)
    sample.non_generation_time = 0.0
    sample.validate()
    return sample


def pad_fanout_samples(samples: list[Sample], alignment: int) -> tuple[list[Sample], int]:
    """Pad a flat fan-out sample list to an ``alignment`` multiple.

    ``alignment`` is dp_size under dynamic-GBS single-step consumption, else
    global_batch_size. Returns ``(samples, pad_count)``; ``samples`` is the
    same list, mutated in place with padding appended after every real segment.
    """
    if not samples or len(samples) % alignment == 0:
        return samples, 0
    padded_size = math.ceil(len(samples) / alignment) * alignment
    pad_count = padded_size - len(samples)
    samples.extend(make_batch_pad_sample(samples[0], pad_index) for pad_index in range(pad_count))
    return samples, pad_count
