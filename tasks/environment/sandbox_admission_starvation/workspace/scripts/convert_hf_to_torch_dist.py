# Adapted from Miles `tools/convert_hf_to_torch_dist.py` (Apache-2.0), adjusted to the
# bundled Miles API. Converts a Hugging Face checkpoint into a Megatron torch_dist
# checkpoint through mbridge and the Miles model plugins.
#
#   torchrun --nproc-per-node N scripts/convert_hf_to_torch_dist.py <megatron model args> \
#       --hf-checkpoint <hf dir> --save <output dir>
import gc
import os
import shutil

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401  (registers the Qwen3.5 bridges)
from mbridge import AutoBridge
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.utils.arguments import get_miles_extra_args_provider
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.memory_utils import print_memory
from miles.utils.hf_config import load_hf_config


def get_args():
    # Full Miles argument set (it already defines --hf-checkpoint and --megatron-to-hf-mode),
    # so miles.backends.megatron_utils.initialize.init() finds every attribute it reads.
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert int(os.environ.get("WORLD_SIZE", "1")) <= args.num_layers
    validate_args(args)
    return args


def main():
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    global_rank = int(os.getenv("RANK") or 0)
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank,
                            device_id=torch.device(f"cuda:{local_rank}"))
    args = get_args()
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    try:
        bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True,
                                            make_vocab_size_divisible_by=args.make_vocab_size_divisible_by)
    except (ValueError, KeyError):
        bridge = AutoBridge.from_config(load_hf_config(args.hf_checkpoint))
    bridge.vocab_size = args.vocab_size
    bridge.padded_vocab_size = args.padded_vocab_size
    enforce_marked_param_dtypes(model)
    bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)
    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    save_checkpoint(1, model, None, None, 0)
    if dist.get_rank() == 0:
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    # Written last: the launcher treats the tracker file as the completion marker.
    if dist.get_rank() == 0:
        with open(get_checkpoint_tracker_filename(args.save), "w") as handle:
            handle.write("release")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
