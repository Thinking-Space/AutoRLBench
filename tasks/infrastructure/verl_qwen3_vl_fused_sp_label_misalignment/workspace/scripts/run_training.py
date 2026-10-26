"""Run the recorded forty-update recipe from a portable workspace."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser()
p.add_argument('--config-only', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / 'framework'
run = root / 'runs/session'
sys.path.insert(0, str(source))
os.environ['PYTHONPATH'] = os.pathsep.join([str(source), str(root/'scripts')])
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
params = [s.replace('/workspace', str(root)) for s in json.loads((root/'config/overrides.json').read_text())]
with initialize_config_dir(config_dir=str(source/'verl/trainer/config'), version_base=None):
    config = compose(config_name='ppo_trainer', overrides=params)
assert config.trainer.n_gpus_per_node == 4
assert config.actor_rollout_ref.actor.fsdp_config.fsdp_size == 4
assert config.actor_rollout_ref.actor.ulysses_sequence_parallel_size == 2
assert config.actor_rollout_ref.model.use_fused_kernels
assert config.actor_rollout_ref.model.use_remove_padding
assert config.trainer.resume_mode == 'disable'
assert config.trainer.total_training_steps == 200
if a.config_only:
    print(OmegaConf.to_yaml(config, resolve=True))
    raise SystemExit(0)
if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 4:
    raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the four allocated devices')
for path in [root/'resources/model/config.json', root/'resources/train.parquet', root/'resources/validation.parquet']:
    if not path.is_file():
        raise FileNotFoundError(path)
if run.exists():
    raise FileExistsError('Preserve existing run output: ' + str(run))
run.mkdir(parents=True)
os.environ.update(BENCH_STAGE_STOP='40', BENCH_GPU_DEADLINE='inf', WANDB_MODE='offline',
                  PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                  GLOO_SOCKET_IFNAME='lo', NCCL_IB_DISABLE='1', NCCL_SHM_DISABLE='1',
                  NCCL_CUMEM_HOST_ENABLE='0', RAY_DEDUP_LOGS='0', HYDRA_FULL_ERROR='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
for key, suffix in [('TMPDIR','tmp'), ('RAY_TMPDIR','ray')]:
    path=root/'runtime'/suffix;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
for key, suffix in [('WANDB_DIR','wandb'), ('WANDB_CONFIG_DIR','wandb/config'),
                    ('WANDB_CACHE_DIR','wandb/cache'), ('WANDB_DATA_DIR','wandb/data')]:
    (run/suffix).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(run/suffix)
for key, suffix in [('HF_HOME','hf'), ('HF_DATASETS_CACHE','datasets'), ('TORCH_HOME','torch'),
                    ('TORCH_EXTENSIONS_DIR','torch_extensions'), ('TRITON_CACHE_DIR','triton'),
                    ('CUDA_CACHE_PATH','cuda'), ('VLLM_CACHE_ROOT','vllm'), ('XDG_CACHE_HOME','cache')]:
    path=root/'runtime'/suffix;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['VERL_FILE_LOGGER_PATH'] = str(run/'scalars.jsonl')
os.environ.pop('RAY_ADDRESS', None)
(run/'overrides.json').write_text(json.dumps(params, indent=2)+'\n')
(run/'resolved.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
sys.argv = ['verl.trainer.main_ppo', *params]
try:
    runpy.run_module('verl.trainer.main_ppo', run_name='__main__')
finally:
    import ray
    ray.shutdown()
