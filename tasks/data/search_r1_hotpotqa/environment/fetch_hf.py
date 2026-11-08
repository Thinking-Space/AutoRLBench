#!/usr/bin/env python3
"""Fetch the task's pinned public assets, cache-first, into --dest.

The CLI follows the task-family fetch_hf.py interface: --repo, --type and
--dest. This configuration also requires --revision and verifies the pinned
file lists, sizes and SHA-256 values. Transfers use HTTPS without Xet; model
weights support resuming an interrupted transfer. Cache keys include revision.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


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

def materialize(source: Path, dest: Path) -> None:
    """Put the cached tree at `dest`, without paying for a second copy if we can.

    Hard links, falling back to a copy across filesystems. The cache roots and
    `--dest` are normally two directories on one mount, so the link path is the
    usual one and a 1 GB checkpoint costs inodes rather than gigabytes.

    Not symlinks: the link target is a container path (/autorlbench/models/...),
    and on the harbor path `--dest` is uploaded into Harbor's container as a
    plain copy of the workspace, where a symlink would dangle. `--dest` has to be
    real bytes the agent and verifier stages can read on their own.

    Any symlink inside `source` is resolved to its real target before linking
    (`os.link` does not follow symlinks on every filesystem, notably the
    Cybertron `/user` mount, and a hardlink to a symlink leaves a dangling entry
    at `dest`). Resolving to the real file keeps `dest` real bytes everywhere.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    def _link_real(src: str, dst: str) -> None:
        os.link(os.path.realpath(src), dst)

    try:
        shutil.copytree(source, dest, copy_function=_link_real, symlinks=False)
    except OSError:
        # Cross-device, or a filesystem without hard links. copytree's default
        # copy_function open()s the source, which always follows symlinks, so
        # the real bytes are copied and this path never dangles.
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, symlinks=False)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def validate_model(folder, model):
    for name in model['files']:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Missing model file: ' + str(path))
        if name.endswith('.json'):
            json.loads(path.read_text())
    weight = folder / 'model.safetensors'
    if weight.stat().st_size != model['bytes'] or sha256(weight) != model['sha256']:
        raise RuntimeError('Model weight size or SHA-256 mismatch: ' + str(weight))

def download_model(folder, model):
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
    validate_model(folder, model)

def prepare_model(args):
    candidates = [model for model in MODELS if model['repo'] == args.repo]
    if len(candidates) != 1 or candidates[0]['revision'] != args.revision:
        raise ValueError('Model repo and revision must match the pinned configuration')
    model = candidates[0]
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
    cache_root = Path(os.environ.get('AUTORLBENCH_MODELS') or workspace / '.model-cache')
    cache_root.mkdir(parents=True, exist_ok=True)
    key = model['repo'].replace('/', '__') + '--' + model['revision']
    cache = cache_root / key
    with (cache_root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.is_dir():
            validate_model(cache, model)
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
            download_model(staging, model)
            (staging / '.complete.json').write_text(json.dumps(model, indent=2))
            pending.unlink()
            staging.rename(cache)
        if args.dest.exists():
            validate_model(args.dest, model)
        else:
            materialize(cache, args.dest)
    prepared = Path(os.environ.get('AUTORLBENCH_PREPARED') or workspace / '.prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    with (prepared / 'models.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record_path = prepared / 'models.json'
        records = json.loads(record_path.read_text()) if record_path.exists() else []
        records = [record for record in records if record['repo'] != model['repo']]
        record = {key: model[key] for key in ['repo', 'revision', 'dest', 'bytes', 'sha256']}
        record['file_sha256'] = {name: sha256(args.dest / name) for name in model['files']}
        records.append(record)
        record_path.write_text(json.dumps(records, indent=2))


def validate_data(path, spec):
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size != spec['bytes']
            or sha256(path) != spec['sha256']):
        raise RuntimeError('Data size or SHA-256 mismatch: ' + str(path))

def download_data(path, spec, source):
    endpoints = list(dict.fromkeys(filter(None, [
        os.environ.get('HF_ENDPOINT', '').rstrip('/'),
        'https://huggingface.co', 'https://hf-mirror.com',
    ])))
    for endpoint in endpoints:
        url = (endpoint + '/datasets/' + source['repo_id'] + '/resolve/'
               + source['revision'] + '/' + spec['remote_path'] + '?download=true')
        for attempt in range(1, 3):
            print('[prepare] downloading', source['repo_id'], spec['remote_path'],
                  'endpoint', endpoint, 'attempt', attempt, flush=True)
            # -q ignores local curl configuration; public downloads use no token.
            result = subprocess.run([
                'curl', '-q', '--fail', '--silent', '--show-error', '--location',
                '--connect-timeout', '15', '--max-time', '180',
                '--output', str(path), url,
            ], capture_output=True, text=True)
            if result.returncode == 0:
                validate_data(path, spec)
                return endpoint
            print('[prepare] download retry, curl_exit', result.returncode, flush=True)
            time.sleep(attempt)
    raise RuntimeError('Public download failed: ' + source['repo_id'] + '/' + spec['remote_path'])

def prepare_data(args):
    manifest = json.loads(args.manifest.read_text())
    source = manifest['source']
    if args.repo != source['repo_id'] or args.revision != source['revision']:
        raise ValueError('Dataset repo and revision must match the data manifest')
    revision = source['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Data revision must be a complete commit hash')
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
    cache_root = Path(os.environ.get('AUTORLBENCH_DATASETS') or workspace / '.dataset-cache')
    cache_root.mkdir(parents=True, exist_ok=True)
    key = source['repo_id'].replace('/', '__') + '--' + revision
    cache = cache_root / key
    downloads = []
    with (cache_root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.is_dir():
            for name, spec in manifest['files'].items():
                validate_data(cache / name, spec)
            print('[prepare] verified data cache', source['repo_id'], flush=True)
        else:
            with tempfile.TemporaryDirectory(prefix=key + '.partial-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'snapshot'
                staging.mkdir()
                for name, spec in manifest['files'].items():
                    path = staging / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    endpoint = download_data(path, spec, source)
                    downloads.append({'file': spec['remote_path'], 'endpoint': endpoint})
                (staging / '.complete.json').write_text(json.dumps(source, indent=2) + '\n')
                staging.rename(cache)
        for name, spec in manifest['files'].items():
            destination = args.dest / Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                validate_data(destination, spec)
                continue
            descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '-', dir=destination.parent)
            os.close(descriptor)
            pending = Path(temporary)
            try:
                shutil.copyfile(cache / name, pending)
                pending.chmod(0o644)
                validate_data(pending, spec)
                os.replace(pending, destination)
            finally:
                pending.unlink(missing_ok=True)
            validate_data(destination, spec)
    prepared = Path(os.environ.get('AUTORLBENCH_PREPARED') or workspace / '.prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / 'data.json').write_text(json.dumps({
        'source': source, 'files': manifest['files'], 'downloads': downloads,
        'anonymous_download': True, 'verified': True,
    }, indent=2) + '\n')
    print('[prepare] public data verified', source['repo_id'], revision, flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--type', choices=['model', 'dataset'], default='model')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('data_manifest.json'))
    args = parser.parse_args(argv)
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('--revision must be a complete commit hash')
    if args.type == 'dataset':
        prepare_data(args)
    else:
        prepare_model(args)


if __name__ == '__main__':
    main()
