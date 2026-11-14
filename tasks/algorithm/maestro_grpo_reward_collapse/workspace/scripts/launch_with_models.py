#!/usr/bin/env python3
"""Start the six local vLLM services, wait for readiness, then run training."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Platform containers route http_proxy through a whitelist proxy that returns
# 502 for localhost; every health check and the training code's OpenAI client
# talk to 127.0.0.1 services, so exempt them before anything is spawned (the
# variable propagates to the service and training subprocesses below).
os.environ['no_proxy'] = (os.environ.get('no_proxy', '')
                          + ',localhost,127.0.0.1').lstrip(',')
from tools.service_config import GPU_MEMORY_LIMIT, get_services
from tools.log_paths import get_run_log_dir


class StopRequested(Exception):
    def __init__(self, signum):
        self.signum = signum


def validate_layout(services, training_gpus):
    training = training_gpus.split(',')
    if len(training) != 4 or len(set(training)) != 4:
        raise ValueError('TRAIN_GPUS must contain four distinct physical GPU IDs')
    lookup = {service.key: service for service in services}
    if lookup['qwen_vl'].gpu != lookup['deepeyes'].gpu:
        raise ValueError('Qwen and DeepEyes must share one inference GPU')
    if lookup['chart'].gpu != lookup['medgemma'].gpu:
        raise ValueError('Chart-R1 and MedGemma must share one inference GPU')
    inference = {service.gpu for service in services}
    if len(inference) != 4 or any(',' in gpu for gpu in inference):
        raise ValueError('The six inference services must occupy exactly four single GPUs')
    if inference.intersection(training):
        raise ValueError('Inference and training GPUs must not overlap')
    if len({service.port for service in services}) != len(services):
        raise ValueError('Model service ports must be distinct')
    memory_by_gpu = defaultdict(float)
    for service in services:
        path = Path(service.model_dir)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Model directory must be project-relative: {path}')
        if not 0 < service.gpu_memory_utilization < 1:
            raise ValueError(f'Invalid memory fraction for {service.name}')
        memory_by_gpu[service.gpu] += service.gpu_memory_utilization
        url = urlsplit(service.base_url)
        if url.scheme != 'http' or url.port != service.port or url.path.rstrip('/') != '/v1':
            raise ValueError(f'Client API URL does not match the deployment port: {service.base_url}')
    for gpu, total in memory_by_gpu.items():
        if total > GPU_MEMORY_LIMIT + 1e-9:
            raise ValueError(f'GPU {gpu} combined inference memory budget {total:.4f} exceeds the {GPU_MEMORY_LIMIT:.2f} limit')


def build_server_command(service, python):
    if service.key == 'glm' and os.environ.get('GLM_PYTHON'):
        interpreter = [os.environ['GLM_PYTHON']]
    elif service.key == 'glm':
        conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
        if not conda or not Path(conda).is_file():
            raise RuntimeError('conda is required for GLM; set CONDA_EXE to its executable')
        interpreter = [conda, 'run', '--no-capture-output', '-n',
                       os.environ.get('GLM_CONDA_ENV', 'vllm_glm_ocr'), 'python']
    else:
        interpreter = [python]
    command = interpreter + [
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', service.model_dir,
        '--served-model-name', service.served_model_name,
        '--host', os.environ.get('VLLM_BIND_HOST', '127.0.0.1'),
        '--port', str(service.port),
        '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
        '--trust-remote-code',
        '--gpu-memory-utilization', str(service.gpu_memory_utilization),
        '--max-model-len', str(service.max_model_len),
        '--max-num-seqs', str(service.max_num_seqs),
        '--limit-mm-per-prompt', '{"image":4}',
    ]
    if service.enforce_eager:
        command.append('--enforce-eager')
    return command


def preflight(services, training_gpus):
    missing = []
    for service in services:
        path = ROOT / service.model_dir
        if not (path / 'config.json').is_file() or not (
            any(path.glob('*.safetensors')) or any(path.glob('pytorch_model*.bin'))
        ):
            missing.append(service.model_dir)
    if missing:
        raise RuntimeError('Place complete model files in these reserved directories first:\n  ' + '\n  '.join(missing))
    executable = shutil.which('nvidia-smi')
    if not executable:
        raise RuntimeError('nvidia-smi is required to verify the eight physical GPUs')
    result = subprocess.run(
        [executable, '--query-gpu=index', '--format=csv,noheader'],
        check=True, text=True, capture_output=True, timeout=15,
    )
    available = {line.strip() for line in result.stdout.splitlines()}
    required = {service.gpu for service in services} | set(training_gpus.split(','))
    if not required.issubset(available):
        raise RuntimeError(f'Required GPUs {sorted(required)}; available GPUs {sorted(available)}')
    # Refuse occupied ports; never kill or reuse unrelated existing services.
    for service in services:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((os.environ.get('VLLM_BIND_HOST', '127.0.0.1'), service.port))
            except OSError as error:
                raise RuntimeError(f'Port {service.port} for {service.name} is unavailable: {error}') from error


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Also collect workers which outlive their original parent process.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class ServiceWatchdog:
    """Restart one owned service without terminating training or its peers."""

    def __init__(self, name, command, env, log, health_url, stop, gpu_lock,
                 model_url=None, model_name=None):
        self.name, self.command, self.env, self.log = name, command, env, log
        self.health_url, self.model_url, self.model_name = health_url, model_url, model_name
        self.stop, self.gpu_lock = stop, gpu_lock
        self.ready = threading.Event()
        self.process = None
        self.restarts = 0
        self.start_timeout = float(os.environ.get('MODEL_START_TIMEOUT', '1800'))
        self.health_interval = float(os.environ.get('MODEL_HEALTH_INTERVAL', '5'))
        self.health_failures = int(os.environ.get('MODEL_HEALTH_FAILURES', '3'))
        if self.start_timeout <= 0 or self.health_interval <= 0 or self.health_failures < 1:
            raise ValueError('Service timeouts/intervals and health failure threshold must be positive')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.thread = threading.Thread(target=self._watch, name=f'watchdog-{name}', daemon=True)

    def report(self, message):
        line = f'[watchdog {self.name}] {message}'
        print(line, flush=True)
        self.log.write('\n' + line + '\n')
        self.log.flush()

    def healthy(self, verify_model=False):
        try:
            with self.opener.open(self.health_url, timeout=3) as response:
                if response.status != 200:
                    return False
            if verify_model and self.model_url:
                with self.opener.open(self.model_url, timeout=3) as response:
                    ids = {item['id'] for item in json.load(response).get('data', [])}
                return self.model_name in ids
            return True
        except (OSError, ValueError, KeyError):
            return False

    def _launch(self):
        # Serialise reloads sharing a GPU, including vLLM memory profiling.
        while not self.stop.is_set():
            if self.gpu_lock.acquire(timeout=0.5):
                break
        else:
            return False
        try:
            if self.stop.is_set():
                return False
            if self.process is not None:
                stop_process_group(self.process)
            self.report(f'Starting (restart #{self.restarts})')
            self.process = subprocess.Popen(
                self.command, cwd=ROOT, env=self.env, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            deadline = time.monotonic() + self.start_timeout
            while not self.stop.is_set() and time.monotonic() < deadline:
                result = self.process.poll()
                if result is not None:
                    raise RuntimeError(f'exited with code {result} before readiness')
                if self.healthy(verify_model=True):
                    self.ready.set()
                    self.report(f'Ready: PID={self.process.pid}, health={self.health_url}')
                    return True
                self.stop.wait(1)
            if not self.stop.is_set():
                raise TimeoutError(f'readiness timed out after {self.start_timeout}s')
            return False
        finally:
            self.gpu_lock.release()

    def _watch(self):
        failed_starts = 0
        while not self.stop.is_set():
            try:
                if not self._launch():
                    break
                failed_starts = 0
                failures = 0
                next_health = time.monotonic() + self.health_interval
                while not self.stop.wait(1):
                    result = self.process.poll()
                    if result is not None:
                        raise RuntimeError(f'process exited with code {result}')
                    if time.monotonic() >= next_health:
                        failures = 0 if self.healthy() else failures + 1
                        next_health = time.monotonic() + self.health_interval
                        if failures >= self.health_failures:
                            raise RuntimeError(f'health check failed {failures} times')
            except Exception as error:
                self.ready.clear()
                self.restarts += 1
                self.report(f'{error}; restarting this service; training stays running')
                # The first reload is immediate. Back off only if reloading fails.
                delay = min(30, 2 ** min(failed_starts, 5)) if failed_starts else 0
                failed_starts += 1
                self.stop.wait(delay)
        self.ready.clear()


def run(services, command, training_gpus, python, tool_host=None, tool_port=None):
    watchdogs = []
    logs = []
    training = None
    stop = threading.Event()
    gpu_locks = defaultdict(threading.Lock)
    log_dir = get_run_log_dir('models') / 'model_services'
    log_dir.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise StopRequested(signum)
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if tool_port is not None:
            log = open(log_dir.parent / 'tool_server.log', 'a', buffering=1)
            logs.append(log)
            watchdogs.append(ServiceWatchdog(
                'tool-server', [python, '-m', 'verl_tool.servers.serve', '--host', tool_host,
                                '--port', str(tool_port), '--tool_type', 'ipython_code',
                                '--workers_per_tool', '8', '--use_ray=True'],
                dict(os.environ, PYTHONUNBUFFERED='1'), log,
                f'http://{tool_host}:{tool_port}/health', stop, threading.Lock(),
            ))
        for service in services:
            log = open(log_dir / (service.key + '.log'), 'x', buffering=1)
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=service.gpu, PYTHONUNBUFFERED='1')
            env.pop('RAY_ADDRESS', None)
            if service.key == 'glm':
                # The GLM venv is self-contained (uv-managed CPython 3.11). Drop
                # the inherited PYTHONPATH: it points the cp312-built pysite
                # shadow at a py3.11 interpreter, where its numpy import fails.
                env.pop('PYTHONPATH', None)
                # Do not share compiled artifacts between the two torch stacks.
                for variable, directory in [('VLLM_CACHE_ROOT', 'vllm-glm'),
                                             ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor-glm'),
                                             ('TRITON_CACHE_DIR', 'triton-glm')]:
                    env[variable] = str(ROOT / '.cache' / directory)
            print(f'Starting {service.name} on GPU {service.gpu}; memory fraction {service.gpu_memory_utilization:.2f}', flush=True)
            watchdogs.append(ServiceWatchdog(
                service.name, build_server_command(service, python), env, log,
                service.base_url[:-3] + '/health', stop, gpu_locks[service.gpu],
                model_url=service.base_url + '/models', model_name=service.served_model_name,
            ))
        # Start sequentially; watch already-started services during the remaining startup.
        for watchdog in watchdogs:
            watchdog.thread.start()
            while not watchdog.ready.wait(0.5):
                if not watchdog.thread.is_alive():
                    raise RuntimeError(f'{watchdog.name} watchdog unexpectedly stopped')
        while not all(watchdog.ready.is_set() for watchdog in watchdogs):
            time.sleep(0.5)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=training_gpus, PYTHONUNBUFFERED='1')
        print(f'All six models ready. Starting training on GPUs {training_gpus}.', flush=True)
        training = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        while training.poll() is None:
            for watchdog in watchdogs:
                if not watchdog.thread.is_alive():
                    raise RuntimeError(f'{watchdog.name} watchdog unexpectedly stopped')
            time.sleep(1)
        result = training.returncode
        return result if result >= 0 else 128 - result
    finally:
        # Ignore repeated signals while terminating only processes created here.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        stop.set()
        if training is not None:
            stop_process_group(training)
        for watchdog in watchdogs:
            if watchdog.thread.ident is not None:
                watchdog.thread.join()
        for watchdog in reversed(watchdogs):
            if watchdog.process is not None:
                stop_process_group(watchdog.process)
        for log in logs:
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    parser.add_argument('--tool-server-host', default='127.0.0.1')
    parser.add_argument('--tool-server-port', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Provide the training command after --')
    requested_url = os.environ.get('TOOL_SERVER_URL', '')
    if requested_url:
        from urllib.parse import urlsplit
        requested = urlsplit(requested_url)
        if requested.hostname not in {'127.0.0.1', 'localhost'} or not requested.port or requested.path.rstrip('/') != '/get_observation':
            raise ValueError('TOOL_SERVER_URL must be a local /get_observation URL with a port')
        args.tool_server_host = requested.hostname
        args.tool_server_port = requested.port
    os.chdir(ROOT)
    services = get_services()
    training_gpus = os.environ.get('TRAIN_GPUS', '4,5,6,7')
    python = os.environ.get('VLLM_PYTHON', sys.executable)
    validate_layout(services, training_gpus)
    for service in services:
        print(f'{service.name}: GPU={service.gpu}, API={service.base_url}, model={service.model_dir}', flush=True)
        if args.dry_run:
            print('  CUDA_VISIBLE_DEVICES=' + service.gpu + ' ' + shlex.join(build_server_command(service, python)), flush=True)
    if args.dry_run:
        print('TRAIN: CUDA_VISIBLE_DEVICES=' + training_gpus + ' ' + shlex.join(command), flush=True)
        return 0
    preflight(services, training_gpus)
    return run(services, command, training_gpus, python, args.tool_server_host, args.tool_server_port)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except StopRequested as error:
        sys.exit(128 + error.signum)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
