# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
from typing import List, Optional

from megatron.core import mpu
from modelbest_sdk.dataset.batch_packer.batch_packer_factory import MEGATRON_BATCH_PACKER
from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader
from modelbest_sdk.dataset.sampler.sampler_factory import WEIGHTED_MEGATRON_SAMPLER
from modelbest_sdk.dataset.segment.segment_factory import CONDITIONAL_FIXED_LENGTH_SEGMENT
from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import DatasetInfo, DatasetInfoList
from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext


class ModelBestSDKDatasetBuilder:
    """Build ModelBest SDK dataloaders for Megatron external dataloader mode."""

    def __init__(self, config: DatasetContext):
        self.config = config

    def build(self, args) -> List[Optional[ModelbestDataloader]]:
        if args.data_path is not None:
            dataloader = self._build(args.data_path, args)
            return dataloader, None, None

        train_dataloader = self._build(args.train_data_path, args) if args.train_data_path else None
        valid_dataloader = self._build(args.valid_data_path, args) if args.valid_data_path else None
        test_dataloader = self._build(args.test_data_path, args) if args.test_data_path else None
        return train_dataloader, valid_dataloader, test_dataloader

    def _build(self, data_path, args) -> ModelbestDataloader:
        dataloader = ModelbestDataloader(
            self.config,
            self.parse_data_path(data_path),
            batch_size=args.micro_batch_size,
            max_len=args.seq_length,
            chunk_size=args.data_chunk_size,
            num_workers=args.num_workers,
            prefetch_chunk_cnt=args.prefetch_chunk_cnt,
            prefetch_factor=args.prefetch_factor,
            prefetch_buffer_size=args.prefetch_buffer_size,
            cache_size=args.data_cache_size_mb * 1024 * 1024,
            cache_budget_mb=args.data_cache_budget_mb,
            worker_recycle_batches=getattr(args, 'data_worker_recycle_batches', 0),
            worker_recycle_sec=getattr(args, 'data_worker_recycle_sec', 0),
            cuda_prefetch=False,
            segment_type=CONDITIONAL_FIXED_LENGTH_SEGMENT,
            sampler_type=WEIGHTED_MEGATRON_SAMPLER,
            batch_packer_type=MEGATRON_BATCH_PACKER,
            dp_group=mpu.get_data_parallel_group(),
            clear_sampler_state=args.clear_sampler_state,
            track_drop_stats=getattr(args, 'track_drop_stats', False),
        )

        if not args.no_load_data_state:
            if args.ckpt_step:
                load_step = args.ckpt_step
            else:
                load_step = os.popen(f"cat {args.load}/latest_checkpointed_iteration.txt").read().strip()
            if load_step == "release":
                dataset_ckpt_path = os.path.join(args.load, f'release')
            else:
                dataset_ckpt_path = os.path.join(args.load, f'iter_{int(load_step):07d}')
            if not os.path.exists(dataset_ckpt_path + "/dataset_ckpt"):
                raise ValueError(f"dataset ckpt directory {dataset_ckpt_path + "/dataset_ckpt"} does not exist ......")
            print(f">>>> loading dataset ckpt from directory {dataset_ckpt_path} ......")
            dataloader.resume(dataset_ckpt_path)

        return dataloader

    @staticmethod
    def normalize(weights):
        total = sum(weights)
        return [w / total for w in weights]

    def parse_data_path(self, data_path):
        weights, paths = zip(
            *[(float(data_path[i]), data_path[i + 1].strip()) for i in range(0, len(data_path), 2)]
        )
        weights = self.normalize(weights)
        return DatasetInfoList(
            [DatasetInfo(path=path, weight=weight) for path, weight in zip(paths, weights)]
        )
