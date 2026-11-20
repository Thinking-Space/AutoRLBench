"""Stdlib helpers using the bundled real dataset and actual task entry points."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / 'tasks/data/justrl2_data_relocation'
ENVIRONMENT = TASK / 'environment'
ENGINE = ENVIRONMENT / 'incident_engine'
REFERENCE_LOADER = TASK / 'evaluations/solution/oracle/reference_loader.py'
VERIFIER = TASK / 'evaluations/solution/verifier'
COPY_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc')


def load(path):
    """Exercise standalone importlib loading without supplying engine sys.path."""
    path = Path(path)
    assert path.is_file(), f'Required production API has not landed: {path}'
    spec = importlib.util.spec_from_file_location('_review_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def inventory(root):
    return {p.relative_to(root).as_posix(): {
        'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def clean_env(**values):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('AUTORLBENCH_', 'RELOCATION_', 'BENCH_RESTORE_', 'PYTHON'))}
    return dict(env, PYTHONDONTWRITEBYTECODE='1',
                **{key: str(value) for key, value in values.items()})


def execute(args, *, env=None, cwd=None, input=None):
    return subprocess.run([str(arg) for arg in args], env=env or clean_env(), cwd=cwd,
                          input=input, capture_output=True, text=True, timeout=45)


class Scene:
    def __init__(self, root):
        source = ENVIRONMENT / 'scene'
        assert (source / 'dataset/manifest.json').is_file(), 'Bundled real dataset has not landed'
        self.scene = root / 'scene'
        shutil.copytree(source, self.scene, ignore=COPY_IGNORE)
        self.snapshot = root / 'snapshot.json'
        shutil.copyfile(ENVIRONMENT / 'snapshot.json', self.snapshot)
        self.dataset = self.scene / 'dataset'
        self.rows = read(self.dataset / 'manifest.json')['rows']
        self.catalog = read(self.dataset / 'catalog.json')['rows']
        self.loader = root / 'loader.py'
        shutil.copyfile(REFERENCE_LOADER, self.loader)

    def check(self, source=None):
        if source is not None:
            self.loader.write_text(source, encoding='utf-8')
        return load(ENGINE / 'check_reads.py').verify_submission(
            self.scene, self.loader, self.snapshot)


def assert_reads_pass(result):
    assert isinstance(result, dict)
    assert result['score'] is True, result
    assert result['errors'] == []
    variants = {entry['name']: entry for entry in result['variants']}
    assert len(result['variants']) == 2 and set(variants) == {'original', 'relocated'}
    for entry in variants.values():
        assert entry['passed'] is True
        assert (entry['readable_count'], entry['easy'], entry['hard']) == (64, 32, 32)
    assert result['checked_records'] == 128
    assert {probe['name'] for probe in result['probes']} == {
        'unknown', 'content', 'unknown_existing', 'content_existing',
        'missing_current_file', 'ambiguous_catalog'}
    assert len({probe['name'] for probe in result['probes']}) == len(result['probes'])
    assert all(isinstance(probe['name'], str) and probe['passed'] is True
               for probe in result['probes'])


def offline_python(root):
    """Run preparation without site packages or Python network/process access."""
    bin_dir = root / 'offline-bin'
    bin_dir.mkdir()
    guard = bin_dir / 'guard.py'
    guard.write_text('''import runpy
import sys

def audit(event, args):
    if event.startswith('socket.') or event in {
        'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'
    }:
        raise RuntimeError('Preparation attempted network or external process: ' + event)

sys.addaudithook(audit)
args = sys.argv[1:]
while args and args[0] in ('-I', '-S', '-B', '-u'):
    args.pop(0)
if not args or args[0].startswith('-'):
    raise RuntimeError('Preparation must invoke its stdlib setup script')
sys.argv = args
sys.path.insert(0, __import__('os').path.dirname(args[0]))
runpy.run_path(args[0], run_name='__main__')
''', encoding='utf-8')
    wrapper = bin_dir / 'python3'
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable)
                       + ' -I -S -B ' + shlex.quote(str(guard)) + ' "$@"\n', encoding='utf-8')
    wrapper.chmod(0o755)
    (bin_dir / 'python').symlink_to(wrapper.name)
    for name in ('curl', 'wget', 'git', 'pip', 'pip3', 'nvidia-smi', 'torchrun'):
        denied = bin_dir / name
        denied.write_text('#!/bin/sh\necho "Unexpected preparation tool: $0" >&2\nexit 97\n')
        denied.chmod(0o755)
    return str(bin_dir) + os.pathsep + os.defpath


class ReviewRun:
    def __init__(self, root, expected='unhealthy', task=TASK):
        root.mkdir(parents=True)
        self.workspace, self.prepared, self.results = (
            root / name for name in ('workspace', 'prepared', 'results'))
        self.environment = root / 'environment'
        self.verifier = root / 'verifier'
        for source, target in ((task / 'environment', self.environment),
                               (task / 'workspace', self.workspace),
                               (task / 'evaluations/solution/verifier', self.verifier)):
            shutil.copytree(source, target, ignore=COPY_IGNORE)
        self.prepared.mkdir()
        self.results.mkdir()
        self.env = clean_env(AUTORLBENCH_ENVIRONMENT=self.environment,
                             AUTORLBENCH_WORKSPACE=self.workspace,
                             AUTORLBENCH_PREPARED=self.prepared,
                             AUTORLBENCH_RESULTS=self.results,
                             PATH=offline_python(root))
        process = execute(['/bin/bash', self.environment / 'prepare.sh'],
                          env=self.env, cwd=self.workspace)
        assert process.returncode == 0, process.stdout + process.stderr
        self.loader = self.workspace / 'loader.py'
        self.report = self.workspace / 'report.md'
        if expected == 'healthy' and task == TASK:
            shutil.copyfile(REFERENCE_LOADER, self.loader)
        write(self.verifier / 'resources/expected_verdict.json',
              {'schema_version': 1, 'expected_verdict': expected})
        self.initial_loader = self.loader.read_bytes()

    def submit(self, reported=None, repair=False):
        if repair:
            shutil.copyfile(REFERENCE_LOADER, self.loader)
        if reported is not None:
            self.report.write_text(f'Verdict: {reported}\n\nChecked the current resolver.\n',
                                   encoding='utf-8')

    def run(self):
        process = execute([sys.executable, '-I', '-S', '-B',
                           self.verifier / 'tests/grade_submission.py'],
                          env=self.env, cwd=self.workspace)
        assert process.returncode in (0, 1), process.stdout + process.stderr
        result = read(self.results / 'result.json')
        assert process.returncode == (0 if result['passed'] else 1), process.stderr
        assert type(result['score']) is int
        assert result['score'] == int(result['passed'])
        assert json.loads(process.stdout) == result
        assert read(self.results / 'metrics.json') == result
        return result
