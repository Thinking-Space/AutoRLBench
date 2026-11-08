"""Fetch pinned model files and materialize them inside the workspace."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from fetch_hf import materialize

MODELS = [
    {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
     'dest': 'actor', 'bytes': 3087467144,
     'sha256': 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee',
     'files': ['config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']},
    {'repo': 'intfloat/e5-base-v2', 'revision': 'f52bf8ec8c7124536f0efb74aca902b2995e5bcd',
     'dest': 'encoder', 'bytes': 437955512,
     'sha256': 'd0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693',
     'files': ['config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']},
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def validate(folder, model):
    for name in model['files']:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Missing model file: ' + str(path))
        if name.endswith('.json'):
            json.loads(path.read_text())
    weight = folder / 'model.safetensors'
    if weight.stat().st_size != model['bytes'] or sha256(weight) != model['sha256']:
        raise RuntimeError('Model weight size or SHA-256 mismatch: ' + str(weight))

def download(folder, model):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    for name in model['files']:
        url = f'{endpoint}/{model["repo"]}/resolve/{model["revision"]}/{name}?download=true'
        path = folder / name
        if path.is_file() and path.stat().st_size:
            if name == 'model.safetensors':
                if path.stat().st_size == model['bytes'] and sha256(path) == model['sha256']:
                    print('[prepare] verified partial-cache file', name, flush=True)
                    continue
            elif not name.endswith('.json') or json.loads(path.read_text()):
                print('[prepare] reusing partial-cache file', name, flush=True)
                continue
        for attempt in range(1, 5):
            command = ['curl', '--fail', '--silent', '--show-error', '--location',
                '--connect-timeout', '20', '--max-time', '1800' if name == 'model.safetensors' else '180',
                '--output', str(path), url]
            if name == 'model.safetensors' and path.exists() and path.stat().st_size:
                command[1:1] = ['--continue-at', '-']
            options = ''
            if token:
                # Keep credentials out of command-line arguments and error messages.
                options = 'header = ' + json.dumps('Authorization: Bearer ' + token) + '\n'
                command[1:1] = ['--config', '-']
            print('[prepare] downloading', model['repo'], name, 'attempt', attempt,
                  'existing_bytes', path.stat().st_size if path.exists() else 0, flush=True)
            result = subprocess.run(command, input=options, text=True)
            if result.returncode == 0:
                break
            if result.returncode == 33 and path.exists():
                path.unlink()  # This server refused Range; retry without a partial file.
            print('[prepare] retry', name, 'curl_exit', result.returncode, flush=True)
            time.sleep(min(attempt * 2, 8))
        else:
            raise RuntimeError(f'Download failed for {model["repo"]}/{name}; partial bytes retained for resumption')
        print('[prepare] downloaded', name, path.stat().st_size, flush=True)
    validate(folder, model)

def main():
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
    cache_root = Path(os.environ.get('AUTORLBENCH_MODELS') or workspace / '.model-cache')
    cache_root.mkdir(parents=True, exist_ok=True)
    records = []
    for model in MODELS:
        key = model['repo'].replace('/', '__') + '--' + model['revision']
        cache = cache_root / key
        with (cache_root / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache.is_dir():
                validate(cache, model)
                print('[prepare] verified cache', model['repo'], flush=True)
            else:
                staging = cache_root / (key + '.partial')
                pending = staging / '.request.json'
                if staging.exists():
                    if not pending.is_file() or json.loads(pending.read_text()) != model:
                        raise RuntimeError('Existing partial cache has a different request: ' + str(staging))
                else:
                    staging.mkdir()
                    pending.write_text(json.dumps(model, indent=2))
                download(staging, model)
                (staging / '.complete.json').write_text(json.dumps(model, indent=2))
                pending.unlink()
                staging.rename(cache)
            destination = workspace / 'shared/models' / model['dest']
            if destination.exists():
                validate(destination, model)
            else:
                materialize(cache, destination)
            records.append({k: model[k] for k in ['repo', 'revision', 'dest', 'bytes', 'sha256']})
    prepared = Path(os.environ.get('AUTORLBENCH_PREPARED') or workspace / '.prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / 'models.json').write_text(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
