"""Run native live generation and PPO in a dedicated three-GPU Ray instance."""
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    import torch
    import ray
    if torch.cuda.device_count() != 3:
        raise ValueError('Expected three dedicated GPUs: actor1 + critic1 + rollout1')
    output = Path(os.environ['RELOCATION_OUTPUT'])
    if (output / 'TRAINING_RETURNED.json').exists() or (output / 'events.jsonl').exists():
        raise ValueError('Use a fresh run output; do not mix independent attempts')
    output.mkdir(parents=True, exist_ok=True)
    from benchmark.sglang_compat import install
    install()
    import train
    sys.argv = ['train.py', *sys.argv[1:]]
    args = train.parse_args()
    if args.debug_train_only or args.debug_rollout_only or args.debug_disable_optimizer or args.load_debug_rollout_data:
        raise ValueError('Real generation and optimizer updates are required')
    if not args.use_critic or args.update_weights_interval != 1:
        raise ValueError('Native PPO with per-step policy refresh is required')
    allowed_env = {key: value for key, value in os.environ.items()
                   if key.startswith(('RELOCATION_', 'NVTE_', 'SGLANG_')) or key in
                   ('BENCH_EVIDENCE_DIR', 'PYTHONPATH', 'OMP_NUM_THREADS', 'CUDA_DEVICE_MAX_CONNECTIONS',
                    'MILES_EXPERIMENTAL_ROLLOUT_REFACTOR', 'TOKENIZERS_PARALLELISM', 'RAY_DEDUP_LOGS',
                    'WANDB_MODE', 'TENSORBOARD_DIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
    args.train_env_vars = {**args.train_env_vars, **allowed_env}
    (output / 'effective_config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')
    ray.init(address='local', num_gpus=3, num_cpus=12,
             object_store_memory=2 * 1024**3, include_dashboard=False,
             _temp_dir=tempfile.mkdtemp(prefix='reloc-ray-', dir='/tmp'),
             runtime_env={'env_vars': allowed_env}, log_to_driver=True)
    start = time.time()
    try:
        asyncio.run(train.train(args))
        from miles.utils.tracking_utils import finish_tracking
        finish_tracking()
        result = {'training_returned': True, 'elapsed_seconds': time.time() - start,
                  'arm': os.environ['RELOCATION_ARM'], 'physical_gpus': 3,
                  'packages': {name: importlib.metadata.version(name)
                               for name in ['torch', 'ray', 'sglang', 'transformers']}}
        (output / 'TRAINING_RETURNED.json').write_text(json.dumps(result, indent=2) + '\n')
        print('RELOCATION_TRAINING_RETURNED ' + json.dumps(result), flush=True)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
