"""Fresh-start two-step smoke or forty-step formal run; never joins attempts."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

from training_config_fourcard import overrides

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['before', 'after'], required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--campaign', type=Path, required=True)
p.add_argument('--stop', type=int, choices=[2, 40], required=True)
p.add_argument('--config-only', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
assert a.campaign.resolve().is_relative_to((root / 'campaigns').resolve())
assert a.run.resolve().is_relative_to(a.campaign.resolve())
source = root / 'src' / a.variant
sys.path.insert(0, str(source))
params = overrides(root, a.run)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

a.run.mkdir(parents=True, exist_ok=True)
with initialize_config_dir(config_dir=str(source / 'verl/trainer/config'), version_base=None):
    config = compose(config_name='ppo_trainer', overrides=params)
(a.run / 'overrides.json').write_text(json.dumps(params, indent=2) + '\n')
(a.run / 'resolved.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
assert config.trainer.n_gpus_per_node == 4
assert config.actor_rollout_ref.actor.fsdp_config.fsdp_size == 4
assert config.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size == 2
assert config.actor_rollout_ref.actor.ulysses_sequence_parallel_size == 2
assert config.data.train_batch_size == 128
assert config.actor_rollout_ref.rollout.n == 8
assert config.actor_rollout_ref.model.use_fused_kernels
assert config.actor_rollout_ref.model.use_remove_padding
assert config.trainer.resume_mode == 'disable'
assert config.trainer.total_training_steps == 200
if a.config_only:
    print('fourcard_config_composed', a.variant, flush=True)
    raise SystemExit(0)
assert json.loads((a.campaign / 'ready.json').read_text())['passed']
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 4
assert int(os.environ['BENCH_STAGE_STOP']) == a.stop
assert float(os.environ['BENCH_GPU_DEADLINE']) > time.time()
assert not (a.run / 'scalars.jsonl').exists(), 'never overwrite a run'
os.environ['WANDB_MODE'] = 'offline'
for key, suffix in [('WANDB_DIR', 'wandb'), ('WANDB_CONFIG_DIR', 'wandb/config'),
                    ('WANDB_CACHE_DIR', 'wandb/cache'), ('WANDB_DATA_DIR', 'wandb/data')]:
    folder = a.run / suffix
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
os.environ['VERL_FILE_LOGGER_PATH'] = str(a.run / 'scalars.jsonl')
os.environ.pop('RAY_ADDRESS', None)
sys.argv = ['verl.trainer.main_ppo', *params]
try:
    runpy.run_module('verl.trainer.main_ppo', run_name='__main__')
finally:
    import ray
    ray.shutdown()
