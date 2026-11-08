"""Download the fixed public data snapshot and verify every input file."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate(path, spec):
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size != spec['bytes']
            or digest(path) != spec['sha256']):
        raise RuntimeError('Data size or SHA-256 mismatch: ' + str(path))


def fetch(path, spec, source):
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
                validate(path, spec)
                return endpoint
            print('[prepare] download retry, curl_exit', result.returncode, flush=True)
            time.sleep(attempt)
    raise RuntimeError('Public download failed: ' + source['repo_id'] + '/' + spec['remote_path'])


def main():
    manifest = json.loads(Path(__file__).with_name('data_manifest.json').read_text())
    source = manifest['source']
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
                validate(cache / name, spec)
            print('[prepare] verified data cache', source['repo_id'], flush=True)
        else:
            with tempfile.TemporaryDirectory(prefix=key + '.partial-', dir=cache_root) as temporary:
                staging = Path(temporary) / 'snapshot'
                staging.mkdir()
                for name, spec in manifest['files'].items():
                    path = staging / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    endpoint = fetch(path, spec, source)
                    downloads.append({'file': spec['remote_path'], 'endpoint': endpoint})
                (staging / '.complete.json').write_text(json.dumps(source, indent=2) + '\n')
                staging.rename(cache)
        for name, spec in manifest['files'].items():
            destination = workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                validate(destination, spec)
                continue
            descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '-', dir=destination.parent)
            os.close(descriptor)
            pending = Path(temporary)
            try:
                shutil.copyfile(cache / name, pending)
                pending.chmod(0o644)
                validate(pending, spec)
                os.replace(pending, destination)
            finally:
                pending.unlink(missing_ok=True)
            validate(destination, spec)
    prepared = Path(os.environ.get('AUTORLBENCH_PREPARED') or workspace / '.prepared')
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / 'data.json').write_text(json.dumps({
        'source': source, 'files': manifest['files'], 'downloads': downloads,
        'anonymous_download': True, 'verified': True,
    }, indent=2) + '\n')
    print('[prepare] public data verified', source['repo_id'], revision, flush=True)


if __name__ == '__main__':
    main()
