"""Run the pinned Ray/FSDP/vLLM experiment, including a real process restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['full', 'before', 'after'])
    parser.add_argument('--uninterrupted', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf
    recipe = OmegaConf.load('configs/experiment.yaml')
    kind = recipe.benchmark.kind
    if args.stage is None:
        stages = ['before', 'after'] if kind == 'configuration' and not args.uninterrupted else ['full']
        for stage in stages:
            subprocess.run([sys.executable, __file__, '--output', str(out), '--stage', stage], check=True)
        (out / 'completed.json').write_text(json.dumps({'completed': True, 'stages': stages}))
        return
    os.environ.update(RUN_OUTPUT=str(out), RUN_STAGE=args.stage, BENCHMARK_KIND=kind,
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', VLLM_USE_V1='0',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', WANDB_MODE='offline')
    if args.stage == 'before':
        os.environ['STOP_AFTER_STEP'] = str(recipe.benchmark.resume_step)
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = OmegaConf.merge(OmegaConf.load('verl/trainer/config/ppo_trainer.yaml'), recipe)
    config.trainer.default_local_dir = str(out / 'checkpoints')
    config.trainer.resume_mode = 'auto' if args.stage == 'after' else 'disable'
    OmegaConf.save(config, out / ('resolved_' + args.stage + '.yaml'))
    import ray
    env = {key: os.environ[key] for key in ['RUN_OUTPUT', 'RUN_STAGE', 'BENCHMARK_KIND',
        'PYTHONDONTWRITEBYTECODE', 'VLLM_USE_V1', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS']}
    env['PYTHONPATH'] = '/workspace'
    if args.stage == 'before':
        env['STOP_AFTER_STEP'] = os.environ['STOP_AFTER_STEP']
    ray.init(num_cpus=4, include_dashboard=False, object_store_memory=134217728,
             runtime_env={'env_vars': env})
    try:
        from verl.trainer.main_ppo import run_ppo
        run_ppo(config)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
