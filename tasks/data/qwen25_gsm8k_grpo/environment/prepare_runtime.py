"""Create a persistent package overlay on the declared base image."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    prepared = Path(os.environ['AUTORLBENCH_PREPARED']).resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    lock_path = Path(__file__).with_name('runtime-lock.json')
    lock = json.loads(lock_path.read_text())
    runtime = prepared / 'runtime'
    python = runtime / 'bin/python'
    marker = runtime / '.complete.json'
    identity = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    current = json.loads(marker.read_text()) if marker.is_file() else {}
    if current.get('lock_sha256') != identity:
        # The base image supplies pip through system site-packages;its ensurepip
        # wheels need not be present in the image.
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(runtime)], check=True)
        packages = [item['name'] + '==' + item['version'] for item in lock['overlay']]
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--upgrade',
                        '--disable-pip-version-check', *packages], check=True)
    # Editable metadata and the actual import root both belong to this workspace.
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                    '--disable-pip-version-check', '-e', str(workspace / 'verl')], check=True)
    check = '''import importlib.metadata as m,json,sys
lock=json.loads(open(sys.argv[1]).read())
for item in lock['overlay']:
 assert m.version(item['name'])==item['version'],(item,m.version(item['name']))
import torch,flash_attn,vllm,transformers,verl
assert torch.__version__=='2.10.0+cu129',torch.__version__
from verl.trainer import main_ppo
from verl.workers.rollout.vllm_rollout import vllm_async_server
print('RUNTIME_READY',json.dumps({n:m.version(n) for n in ['torch','vllm','transformers','ray','verl']}))
'''
    env = {**os.environ, 'PYTHONPATH': str(workspace / 'verl'), 'PYTHONDONTWRITEBYTECODE': '1'}
    subprocess.run([str(python), '-c', check, str(lock_path)], env=env, check=True)
    marker.write_text(json.dumps({'lock_sha256': identity}) + '\n')
    # The framework reads these as data and applies them to the later containers.
    values = {
        'PATH': str(runtime / 'bin') + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'VIRTUAL_ENV': str(runtime), 'RL_PYTHON': str(python),
        'PYTHONPATH': str(workspace / 'scripts') + ':' + str(workspace / 'verl'),
        'PYTHONDONTWRITEBYTECODE': '1', 'WANDB_MODE': 'offline', 'TOKENIZERS_PARALLELISM': 'false',
    }
    (prepared / 'env').write_text(''.join(key + '=' + value + '\n' for key, value in values.items()))


if __name__ == '__main__':
    main()
