import itertools
import logging

from miles.ray.rollout.batch_padding import pad_fanout_samples
from miles.utils.multi_lora import is_multi_lora_enabled

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config):
    metadata = {}

    # Multi-LoRA: record group boundaries (heterogeneous per-adapter group sizes)
    # and lift the collection loop's batch-level step decision out of sample metadata,
    # both before flattening.
    if is_multi_lora_enabled(args) and isinstance(data[0], list):
        metadata["prompt_group_sizes"] = [_nested_sample_count(group) for group in data]
        head = _first_sample(data[0])
        metadata["step_slots"] = list(head.metadata.pop("step_slots", []))
        metadata["step_adapter_names"] = list(head.metadata.pop("step_adapter_names", []))

    # flatten the data if it is a list of lists
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

    # Fan-out (multi-segment) rollouts emit a variable number of samples, so
    # the trim below would drop tail segments — and the anchor (the segment
    # carrying the trajectory's terminal reward) is each trajectory's LAST
    # segment, exactly where the trim cuts. Pad with zero-loss samples instead
    # so every real segment reaches training. See batch_padding.py.
    has_fanout = any((sample.metadata or {}).get("parent_traj_id") is not None for sample in data)
    if has_fanout:
        # Padding target: with dynamic GBS the whole padded batch is consumed
        # in one optimizer step, so the only hard divisibility requirement left
        # is the equal DP split — pad to dp_size (≤ dp-1 pads) instead of
        # global_batch_size (≤ GBS-1 pads), keeping the loss normalizer close
        # to the true segment count. The dp-size target additionally needs
        # token-packing micro-batching (--use-dynamic-batch-size): a fixed
        # micro_batch_size would also require per-rank divisibility by it.
        if args.use_dynamic_global_batch_size and args.use_dynamic_batch_size:
            alignment = train_parallel_config["dp_size"]
        else:
            alignment = args.global_batch_size
        if len(data) % alignment != 0:
            original_size = len(data)
            data, pad_count = pad_fanout_samples(data, alignment)
            logger.info(
                f"pad fan-out samples from {original_size} to {len(data)} (alignment {alignment}); "
                f"all {original_size} real segments are retained"
            )

    if not args.disable_rollout_trim_samples:
        global_batch_size = args.global_batch_size
        if args.use_dynamic_global_batch_size:
            logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
            # A padded fan-out batch must be consumed in one optimizer step;
            # otherwise a static GBS would spread the padding across extra updates.
            dynamic_global_batch_size = (
                len(data)
                if has_fanout
                else _compute_dynamic_global_batch_size(
                    args, train_parallel_config=train_parallel_config, num_samples=len(data)
                )
            )
            metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
            global_batch_size = dynamic_global_batch_size

        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            if trim_len == 0:
                raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
            origin_data_length = len(data)
            data = data[:trim_len]
            logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata


def _first_sample(group):
    return _first_sample(group[0]) if isinstance(group[0], list) else group[0]


def _nested_sample_count(group) -> int:
    if not isinstance(group, list):
        return 1
    return sum(_nested_sample_count(item) for item in group)


def _compute_dynamic_global_batch_size(args, train_parallel_config, num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size

    if is_multi_lora_enabled(args):
        # Batches take groups in multiples of each adapter's
        # min_groups_per_dp_split, so this holds by construction; a violation
        # means a generate fn's group shape broke the invariant.
        if num_samples % dp_size != 0:
            raise ValueError(
                f"Multi-LoRA batch of {num_samples} samples is not divisible by dp_size={dp_size}; "
                "the min_groups_per_dp_split invariant was violated (variable-size generate fn output?)"
            )
        return num_samples

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps=1, wasted={wasted})"
        )

    return dynamic_gbs
