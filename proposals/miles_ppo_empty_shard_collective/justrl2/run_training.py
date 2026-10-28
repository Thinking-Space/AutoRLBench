#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Own an isolated Ray instance and execute public train.py in this process."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    # BENCH_RUN_DIR is unique and must not contain prior training evidence.
    output = Path(os.environ["BENCH_RUN_DIR"]).resolve()
    for name in ["actor", "critic", "evidence", "TRAINING_RETURNED.json"]:
        if (output / name).exists():
            raise RuntimeError("Refuse to mix this run with previous outputs: " + name)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path.cwd()))
    os.environ["RAY_DEDUP_LOGS"] = "0"
    import torch
    import ray
    if torch.cuda.device_count() != 8:
        raise RuntimeError("The fixed actor4+critic4 topology requires exactly 8 visible GPUs")
    tmp = tempfile.mkdtemp(prefix="justrl2-bench-ray-", dir="/tmp")
    started = time.time()
    # Starting a local private Ray instance avoids touching any other instance.
    # All descendants run within this task/container; nothing connects to an
    # existing cluster address.
    ray.init(address="local", num_gpus=8, num_cpus=min(32, os.cpu_count() or 8),
             object_store_memory=2 * 1024**3, include_dashboard=False,
             _temp_dir=tmp, log_to_driver=True)
    try:
        argv = sys.argv[1:]
        if argv and argv[0] == "--":
            argv.pop(0)
        sys.argv = ["train.py", *argv]
        import asyncio
        import train
        from benchmark.sglang_compat import install
        install()
        args = train.parse_args()
        args.train_env_vars = {
            **args.train_env_vars,
            **{key: value for key, value in os.environ.items()
               if key in ("BENCH_EVIDENCE_DIR", "PYTHONPATH", "OMP_NUM_THREADS",
                          "CUDA_DEVICE_MAX_CONNECTIONS", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_FLASH_ATTN",
                          "TORCH_NCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_DUMP_ON_TIMEOUT",
                          "TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "NCCL_DEBUG")},
        }
        if not args.debug_train_only or not args.use_critic or args.debug_disable_optimizer:
            raise RuntimeError("Expected real actor+critic replay training with optimizer enabled")
        if args.balance_data or args.use_dynamic_batch_size or args.use_tis:
            raise RuntimeError("Case requires the frozen unbalanced replay path")
        (output / "effective_config.json").write_text(json.dumps({
            k: getattr(args, k, None) for k in ["advantage_estimator", "context_parallel_size",
            "actor_num_gpus_per_node", "critic_num_gpus_per_node", "global_batch_size",
            "micro_batch_size", "num_rollout", "gae_lambda_k", "normalize_advantages",
            "debug_train_only", "debug_disable_optimizer", "balance_data", "use_tis",
            "lr", "critic_lr", "seed", "distributed_timeout_minutes"]}, indent=2) + "\n")
        asyncio.run(train.train(args))
        (output / "TRAINING_RETURNED.json").write_text(json.dumps({
            "train_coroutine_returned": True, "elapsed_seconds": time.time() - started,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version()}, indent=2) + "\n")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
