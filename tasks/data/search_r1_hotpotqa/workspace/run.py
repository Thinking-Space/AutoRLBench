"""Check assets or run the recorded Qwen2.5 / HotpotQA configuration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent

def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-name', default=time.strftime('run-%Y%m%d-%H%M%S'))
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    if not args.run_name or not all(c.isalnum() or c in '_-' for c in args.run_name):
        parser.error('Invalid run name')
    config = json.loads((ROOT / 'config.json').read_text())
    if args.steps is not None:
        if args.steps <= 0:
            parser.error('steps must be positive')
        config['steps'] = args.steps
    data = ROOT / 'shared/data'
    required = [data / name for name in ['train.parquet', 'test.parquet', 'catalog_001.jsonl', 'catalog_002.jsonl', 'catalog_001.index', 'catalog_002.index']]
    required += [ROOT / 'shared/models' / name / 'model.safetensors' for name in ['actor', 'encoder']]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Missing assets: ' + ', '.join(missing))
    out = (args.runs_dir.absolute() if args.runs_dir else ROOT / 'runs') / args.run_name
    env = dict(os.environ,
        PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'Search-R1'), PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='0', SEARCH_R1_SHORT_GRPO='1', TRAIN_PYTHON=sys.executable,
        BASE_MODEL=str(ROOT / 'shared/models/actor'), TRAIN_DATA_DIR=str(data), TEST_DATA_DIR=str(data),
        EXPERIMENT_NAME=args.run_name, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT / '.cache/hf'), HF_DATASETS_CACHE=str(ROOT / '.cache/datasets'),
        XDG_CACHE_HOME=str(ROOT / '.cache'), TRITON_CACHE_DIR=str(ROOT / '.cache/triton'),
        TORCH_HOME=str(ROOT / '.cache/torch'), WANDB_MODE='offline', WANDB_DIR=str(out / 'wandb'),
        WANDB_CACHE_DIR=str(ROOT / '.cache/wandb'), WANDB_CONFIG_DIR=str(ROOT / '.cache/wandbconfig'),
        WANDB_DATA_DIR=str(out / 'wandb-data'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', RAY_ADDRESS='local', RAY_TMPDIR=str(Path('/tmp') / ('qh-' + args.run_name[:24])),
        HYDRA_FULL_ERROR='1', NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1')
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        env.pop(key, None)
    url = f'http://127.0.0.1:{config["port"]}'
    batch = config['batch_size']
    overrides = [f'data.train_batch_size={batch}', 'data.val_batch_size=16',
        'data.max_start_length=512', 'data.max_prompt_length=3072', 'data.max_response_length=256',
        'data.max_obs_length=768', 'actor_rollout_ref.rollout.n_agent=4',
        'actor_rollout_ref.rollout.max_num_seqs=32', f'actor_rollout_ref.actor.ppo_mini_batch_size={batch * 4}',
        f'actor_rollout_ref.actor.optim.lr={config["learning_rate"]}', 'actor_rollout_ref.actor.entropy_coeff=0.001',
        '+actor_rollout_ref.actor.checkpoint_dtype=bfloat16', '+actor_rollout_ref.rollout.engine_seed=42',
        '+trainer.seed=42', '+trainer.answer_only_reward=true', '+trainer.ray_num_cpus=8',
        '+trainer.ray_object_store_memory=2147483648', f'trainer.total_training_steps={config["steps"]}',
        f'trainer.total_epochs={math.ceil(config["steps"] / (1024 // batch))}',
        f'trainer.test_freq={config["eval_every"]}', 'trainer.val_before_train=true', 'trainer.val_only=false',
        'trainer.save_freq=-1', 'trainer.logger=[console]', f'trainer.default_local_dir={ROOT / "checkpoints" / args.run_name}',
        f'+trainer.rollout_log_dir={out / "rollouts"}',
        f'retriever.url={url}/{config["training_catalog"]}/training/retrieve',
        f'+retriever.validation_url={url}/{config["validation_catalog"]}/validation/retrieve']
    command = ['bash', str(ROOT / 'Search-R1/train_grpo.sh'), *overrides]
    if args.check:
        code = ('import pathlib, verl, pyarrow.parquet as pq, faiss; '
                f'assert pathlib.Path(verl.__file__).is_relative_to({str(ROOT / "Search-R1")!r}); '
                f'assert pq.ParquetFile({str(data / "train.parquet")!r}).metadata.num_rows == 1024; '
                f'assert pq.ParquetFile({str(data / "test.parquet")!r}).metadata.num_rows == 128; '
                'print("CHECK_OK", verl.__file__)')
        subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, check=True)
        print(json.dumps({'config': config, 'command': command}, indent=2))
        return
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', config['port']))
    out.mkdir(parents=True, exist_ok=False)
    for key in ['WANDB_DIR', 'WANDB_DATA_DIR', 'RAY_TMPDIR']:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps({'config': config, 'command': command,
        'start_time': time.time(), 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob('*.py') if '.cache' not in p.parts and 'runs' not in p.parts}}, indent=2))
    server = trainer = None
    code = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / 'retrieval_server.log').open('w') as stream:
            server = subprocess.Popen([sys.executable, '-u', str(ROOT / 'serve.py'), '--port', str(config['port']),
                '--trace', str(out / 'retrieval.jsonl')], cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Retrieval service exited; inspect its log')
            try:
                with opener.open(url + '/health', timeout=2):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError('Retrieval readiness timeout')
        with (out / 'train.log').open('w') as stream:
            trainer = subprocess.Popen(command, cwd=ROOT / 'Search-R1', env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True)
            for line in trainer.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = trainer.wait()
    finally:
        stop(trainer)
        stop(server)
        (out / 'exit_status.json').write_text(json.dumps({'exit_code': code, 'end_time': time.time()}))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
