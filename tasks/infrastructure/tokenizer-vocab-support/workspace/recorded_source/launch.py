"""Submit a recipe and run official synchronous verl with observable startup."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
import json
from pathlib import Path
import resource
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path("/workspace")
PYTHON = ROOT / ".venv/bin/python"
sys.path.insert(0, str(ROOT))


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def install_cost_tracking():
    """Defer verl imports until Ray has assigned the actor's visible GPUs.

    Ray runs setup hooks before assigning GPUs to an actor. Importing verl here
    calls torch.cuda.is_available(), which freezes CUDA's initial device mapping.
    Observe Tracking's normal import instead, after Ray installs that mapping.
    """
    import importlib.abc
    import importlib.util

    if 'verl.utils.tracking' in sys.modules:
        _patch_cost_tracking()
        return

    class TrackingLoader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            _patch_cost_tracking()

    class TrackingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'verl.utils.tracking':
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            spec.loader = TrackingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, TrackingFinder())


def _patch_cost_tracking():
    """Add allocation and stage GPU-hours to official tracking.

    Only observes metrics; does not change optimization or sampling. The hook is
    imported from the frozen launch.py, using the run directory on PYTHONPATH.
    """
    from verl.utils.tracking import Tracking

    if getattr(Tracking, '_allocation_costs_installed', False):
        return
    original_init, original_log, original_finish = Tracking.__init__, Tracking.log, Tracking.finish

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _patch_group_metrics()
        config = kwargs.get('config')
        if config is None and len(args) > 3:
            config = args[3]
        run = Path(config['trainer']['default_local_dir']).parent
        allocation = json.loads((run / 'allocation.json').read_text())
        self._allocation_costs = {
            'run': run, 'allocation': allocation, 'last_step': -1,
            'stage_seconds': {}, 'training_responses': 0, 'training_output_tokens': 0,
            'responses_per_step': config['data']['train_batch_size'] * config['actor_rollout_ref']['rollout']['n'],
        }

    def save(self, data=None, step=None):
        state = self._allocation_costs
        allocation = state['allocation']
        if data is not None and 'training/global_step' in data and step > state['last_step']:
            state['last_step'] = step
            for key, value in data.items():
                if key.startswith('timing_s/'):
                    name = key.removeprefix('timing_s/')
                    state['stage_seconds'][name] = state['stage_seconds'].get(name, 0.0) + float(value)
            state['training_responses'] += state['responses_per_step']
            state['training_output_tokens'] += data['response_length/mean'] * state['responses_per_step']
        gpus = allocation['gpus']
        seconds = max(0.0, time.time() - allocation['started_at_unix'])
        stages = state['stage_seconds']
        critic_seconds = stages.get('values', 0.0) + stages.get('update_critic', 0.0)
        metrics = {
            'cost/allocated_gpu_hours': gpus * seconds / 3600,
            'cost/training_gpu_hours': gpus * stages.get('step', 0.0) / 3600,
            'cost/critic_gpu_hours': gpus * critic_seconds / 3600,
            'cost/critic_fraction_of_training': critic_seconds / stages['step'] if stages.get('step') else 0.0,
            'cost/training_responses': state['training_responses'],
            'cost/training_output_tokens': state['training_output_tokens'],
        }
        if state['training_output_tokens']:
            metrics['cost/allocated_gpu_hours_per_million_output_tokens'] = (
                metrics['cost/allocated_gpu_hours'] * 1e6 / state['training_output_tokens'])
        report = {
            'allocation': allocation, 'last_completed_step': state['last_step'],
            'measured_at': datetime.now().isoformat(timespec='seconds'),
            'metrics': metrics, 'stage_seconds': stages,
            'definition': 'Allocated GPUs * wall time since Slurm StartTime; includes startup, evaluation, saving and idle waits; excludes queue. Critic GPU-hours attribute both allocated GPUs to value inference and critic update wall time, not CUDA kernel active time.',
        }
        destination = state['run'] / 'costs.json'
        temporary = destination.with_suffix('.json.tmp')
        try:
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(destination)
        except OSError:
            pass  # bookkeeping only; a transient shared-filesystem error must not kill training
        return metrics

    def record(self, data, step, backend=None):
        metrics = save(self, data, step)
        original_log(self, data={**data, **metrics}, step=step, backend=backend)

    def finish(self, exit_code=0):
        if hasattr(self, '_allocation_costs') and not getattr(self, '_finished', False):
            save(self)
        original_finish(self, exit_code=exit_code)

    Tracking.__init__, Tracking.log, Tracking.finish = initialize, record, finish
    Tracking._allocation_costs_installed = True


def _patch_group_metrics():
    """Log how many prompt groups give GRPO no signal (all 8 responses tie).

    Observation only. Called from Tracking.__init__, when the V1 trainer module
    is fully imported; rollout keys are '{prompt uid}_{session}_{index}'.
    """
    trainer = sys.modules['verl.trainer.ppo.v1.trainer_base'].PPOTrainer
    if getattr(trainer, '_group_metrics_installed', False):
        return
    original = trainer._compute_metrics

    def compute(self, batch, metrics, *args, **kwargs):
        original(self, batch, metrics, *args, **kwargs)
        import transfer_queue as tq
        keep = [i for i, tag in enumerate(batch.tags) if not tag.get('is_padding', False)]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=['rm_scores'])
        scores = data['rm_scores'].sum(dim=1).tolist()
        groups = {}
        for i in keep:
            groups.setdefault(batch.keys[i].rsplit('_', 2)[0], []).append(scores[i])
        count = len(groups)
        all_correct = sum(all(score > 0 for score in values) for values in groups.values())
        all_wrong = sum(all(score <= 0 for score in values) for values in groups.values())
        metrics.update({
            'group/count': count,
            'group/all_correct_fraction': all_correct / count,
            'group/all_wrong_fraction': all_wrong / count,
            'group/zero_variance_fraction': sum(len(set(values)) == 1 for values in groups.values()) / count,
        })

    trainer._compute_metrics = compute
    trainer._group_metrics_installed = True


def report_costs(run):
    """Account for an existing run from Slurm and its official console metrics."""
    job_id = (run / 'slurm_job_id.txt').read_text().strip().split(';')[0]
    rows = subprocess.check_output([
        'sacct', '-j', job_id, '-X', '-n', '-P',
        '--format=JobIDRaw,Start,End,ElapsedRaw,State,AllocTRES',
    ], text=True).strip().splitlines()
    job = next(row.split('|') for row in rows if row.split('|')[0] == job_id)
    _, start, end, elapsed, status, resources = job
    gpus = int(re.search(r'(?:^|,)gres/gpu=(\d+)', resources).group(1))
    stages, completed, output_tokens = {}, set(), 0.0
    from omegaconf import OmegaConf
    config = OmegaConf.load(run / 'resolved_config.yaml')
    responses_per_step = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    pattern = r'([\w/.-]+):([-+\d.eE]+)(?=\s+-\s+|\x1b|$)'
    for line in (run / 'train.log').read_text(errors='replace').splitlines():
        match = re.search(r'\bstep:([1-9]\d*) - ', line)
        if not match or int(match[1]) in completed:
            continue
        metrics = {key: float(value) for key, value in re.findall(pattern, line)}
        if 'response_length/mean' not in metrics or 'timing_s/step' not in metrics:
            continue
        completed.add(int(match[1]))
        output_tokens += metrics['response_length/mean'] * responses_per_step
        for key, value in metrics.items():
            if key.startswith('timing_s/'):
                name = key.removeprefix('timing_s/')
                stages[name] = stages.get(name, 0.0) + value
    report = {
        'job_id': job_id, 'state': status, 'started_at': start, 'ended_at': end,
        'gpus': gpus, 'allocation_seconds': int(elapsed),
        'last_completed_step': max(completed, default=0), 'completed_steps': len(completed),
        'training_responses': len(completed) * responses_per_step,
        'training_output_tokens': output_tokens, 'stage_seconds': stages,
        'allocated_gpu_hours': gpus * int(elapsed) / 3600,
        'training_gpu_hours': gpus * stages.get('step', 0.0) / 3600,
        'critic_gpu_hours': gpus * (stages.get('values', 0.0) + stages.get('update_critic', 0.0)) / 3600,
        'definition': 'Slurm allocated GPUs * ElapsedRaw / 3600; includes startup, evaluation, saving and idle waits; excludes queue. Stage totals cover completed console steps only.',
    }
    if output_tokens:
        report['allocated_gpu_hours_per_million_output_tokens'] = report['allocated_gpu_hours'] * 1e6 / output_tokens
    # Keep this Slurm reconciliation separate from the live logger's costs.json.
    destination = run / 'slurm_costs.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


@contextmanager
def stage(name):
    """Report long waits, including the main thread's actual Python location."""
    started = time.monotonic()
    thread_id = threading.get_ident()
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            frame = sys._current_frames().get(thread_id)
            location = (f"{frame.f_code.co_filename}:{frame.f_lineno} ({frame.f_code.co_name})"
                        if frame else "unavailable")
            usage = resource.getrusage(resource.RUSAGE_SELF)
            log(f"WAIT {name}; elapsed={time.monotonic() - started:.0f}s; pid={os.getpid()}; "
                f"cpu_total={usage.ru_utime + usage.ru_stime:.1f}s; "
                f"peak_rss={usage.ru_maxrss / 1024:.0f}MiB; main_thread={location}")

    log(f"START {name}; pid={os.getpid()}")
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        yield
    except BaseException as exc:
        log(f"FAILED {name}; elapsed={time.monotonic() - started:.1f}s; {type(exc).__name__}")
        raise
    else:
        log(f"DONE {name}; elapsed={time.monotonic() - started:.1f}s")
    finally:
        done.set()
        monitor.join()


def watch_for_engine_failure(log_path, process):
    """Stop the driver when a vLLM worker dies; the trainer otherwise waits forever on an idle GPU."""
    patterns = ('Worker failed with error', 'EngineDeadError')
    with log_path.open(errors='replace') as stream:
        stream.seek(0, os.SEEK_END)
        while process.poll() is None:
            line = stream.readline()
            if not line:
                time.sleep(5)
            elif any(pattern in line for pattern in patterns):
                log('Watchdog: a vLLM worker failed; terminating the driver so Slurm marks the job failed')
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                return


def main(experiment):
    # An absolute Python executable does not activate the venv's CLI tools.
    # FlashInfer invokes ninja by name in spawned Ray/vLLM workers.
    os.environ['PATH'] = f"{ROOT / '.venv/bin'}:{os.environ.get('PATH', '/usr/bin:/bin')}"
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    config_dir = ROOT / 'experiments' / experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--time-limit", help="Slurm wall time; defaults to 30 days for JustRL or 4 hours for the GSM8K PPO smoke test")
    parser.add_argument("--check", action="store_true", help="Resolve and validate configuration without submitting")
    parser.add_argument("--costs", action="store_true", help="Reconcile an existing --run-id with Slurm GPU-hours")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--trainer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mem", default="128G", help="Slurm host memory; 32B with CPU offload needs far more")
    parser.add_argument("--gpus", type=int, help="GPUs to request (default 2 for JustRL, 1 for the GSM8K smoke test)")
    parser.add_argument("--private-shm", action="store_true", help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args()
    prefix = 'ppo-gsm8k' if experiment == 'ppo_gsm8k' else 'justrl-math'
    run_id = args.run_id or datetime.now().strftime(prefix + "-%Y%m%d-%H%M%S")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run ID must be a single directory name")
    run = config_dir / 'outputs' / run_id
    gpus = args.gpus or (2 if experiment == 'justrl_math' else 1)
    if args.gpus and not args.worker and not args.trainer:
        overrides.append(f'trainer.n_gpus_per_node={gpus}')
    gpu_args = ['--gpus', str(args.gpus)] if args.gpus else []
    cpus = 16
    if args.costs:
        if not args.run_id:
            parser.error('--costs requires --run-id')
        report_costs(run)
        return
    if args.check:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name='config', overrides=overrides)
        OmegaConf.resolve(config)
        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        print('Configuration checks passed')
        return
    if args.trainer:
        # Keep upstream's Hydra entry point and training behavior. This wrapper
        # adds visibility during imports and blocking calls without patching verl.
        with stage("import official PPO trainer dependencies"):
            from verl.trainer.main_ppo import main as ppo_main
        sys.argv = ["verl.trainer.main_ppo", "--config-path", str(run),
                    "--config-name", "recipe", *overrides]
        log("SwanLab is initialized by verl inside trainer.fit(); no experiment exists before that stage")
        with stage("official PPO entry point (Ray / models / validation / training)"):
            ppo_main()
        return
    if not args.worker:
        run.mkdir(parents=True, exist_ok=False)
        (run / "overrides.txt").write_text("\n".join(overrides) + "\n")
        for source, name in [(config_dir / "config.yaml", "recipe.yaml"),
                             (config_dir / "train.py", "train.py"),
                             (ROOT / 'scripts/training/launch.py', 'launch.py'),
                             (config_dir / "inputs/manifest.json", "inputs.json"),
                             (ROOT / "environment/records/source_pin.json", "environment.json"),
                             (ROOT / "environment/records/ray_runtime_patch.json", "ray_patch.json"),
                             (ROOT / "environment/records/transformers_runtime_patch.json", "transformers_patch.json")]:
            shutil.copyfile(source, run / name)
        command = [str(PYTHON), "-u", str(run / "train.py"), "--worker", "--run-id", run_id, *gpu_args, *overrides]
        if experiment == 'justrl_math':
            shutil.copyfile(ROOT / 'environment/records/runtime_image.json', run / 'runtime_image.json')
            shutil.copyfile(config_dir / 'reward.py', run / 'reward.py')
            shutil.copyfile(config_dir / 'prepare.py', run / 'prepare.py')
            shutil.copyfile(config_dir / 'inputs/template_check.json', run / 'template_check.json')
            command = ['/usr/bin/python3', str(ROOT / 'environment/image.py'), 'run', '--', *command]
        job = subprocess.check_output([
            "sbatch", "--parsable", f"--job-name=verl-{run_id}", "--partition=gpu", "--exclude=compute-node-reserved",
            "--export=ALL,TMPDIR=/tmp", "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={args.mem}", f"--gres=gpu:{gpus}",
            f"--time={args.time_limit or ('30-00:00:00' if experiment == 'justrl_math' else '04:00:00')}",
            f"--chdir={ROOT}", f"--output={run / 'train.log'}",
            "--wrap", shlex.join(command),
        ], text=True).strip()
        (run / "slurm_job_id.txt").write_text(job + "\n")
        print(f"Slurm job: {job}\nRun: {run}")
        return

    job_id = os.environ["SLURM_JOB_ID"]
    log(f"Slurm job={job_id}; host={socket.gethostname()}; run={run}; python={sys.executable}")
    job_info = subprocess.check_output(['scontrol', 'show', 'job', job_id, '-o'], text=True)
    started_at = re.search(r'\bStartTime=(\S+)', job_info).group(1)
    (run / 'allocation.json').write_text(json.dumps({
        'job_id': job_id, 'gpus': gpus, 'host': socket.gethostname(),
        'physical_gpu_ids': os.environ.get('SLURM_JOB_GPUS'),
        'started_at': started_at, 'started_at_unix': datetime.fromisoformat(started_at).timestamp(),
    }, indent=2) + '\n')
    with stage("check node POSIX shared memory"):
        from multiprocessing.shared_memory import SharedMemory
        try:
            memory = SharedMemory(create=True, size=1024)
        except PermissionError:
            if args.private_shm:
                raise
            log("Host /dev/shm is not writable; restart this job in a private mount namespace with an 8 GiB tmpfs")
            command = [
                "unshare", "--user", "--map-root-user", "--mount", "--propagation", "private",
                "bash", "-c",
                'mount -t tmpfs -o size=8G,mode=1777,nosuid,nodev tmpfs /dev/shm && exec "$@"',
                "verbalrl-shm", str(PYTHON), "-u", str(run / "train.py"),
                "--worker", "--private-shm", "--run-id", run_id, *gpu_args, *overrides,
            ]
            os.execvp(command[0], command)
        try:
            memory.buf[:4] = b"verl"
        finally:
            memory.close()
            memory.unlink()
        log(f"POSIX shared memory writable; /dev/shm mode={Path('/dev/shm').stat().st_mode & 0o7777:04o}")
    if args.private_shm:
        with stage("check CUDA and FlashAttention in private mount namespace"):
            ninja = shutil.which('ninja')
            if ninja != str(ROOT / '.venv/bin/ninja'):
                raise RuntimeError(f'Expected the pinned environment ninja, found {ninja}')
            subprocess.run([ninja, '--version'], check=True)
            with (run / "gpu_check.json").open("w") as report:
                subprocess.run([str(PYTHON), "-u", str(ROOT / "environment/checks/gpu.py")],
                               stdout=report, check=True)
            log(f"Allocated GPU check passed; report={run / 'gpu_check.json'}")
    # The upstream logger reads its API key from the environment. Paths and
    # training settings are explicit configuration or command arguments.
    os.environ["SWANLAB_API_KEY"] = (ROOT / ".swanlab").read_text().strip()
    os.environ["SWANLAB_LOG_DIR"] = str(run / "swanlog")
    node_ip = socket.gethostbyname(socket.gethostname())
    os.environ["NO_PROXY"] = f"localhost,127.0.0.1,{node_ip},{socket.gethostname()}"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    with stage("load configuration and Ray launcher dependencies"):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from environment.ray_cluster import local_ray_cluster
    log(f"Starting Ray; CLI log={run / 'ray_start.log'}; node logs=/tmp/verbalrl-{job_id}/session_latest/logs")
    # Per-GPU utilization and memory every 2 s; stops with the Slurm job.
    subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used', '--format=csv',
                      '-l', '2', '-f', str(run / 'gpu_usage.csv')])
    with local_ray_cluster(cpus=cpus, gpus=gpus, directory=Path("/tmp") / f"verbalrl-{job_id}",
                           log_path=run / "ray_start.log", object_store_memory=2 * 1024**3) as address:
        log(f"Ray head ready at {address}; PPO driver and workers still need to initialize")
        overrides = [*overrides, f"trainer.experiment_name={run_id}", f"ray_kwargs.ray_init.address={address}"]
        # Ray's public setup hook installs metrics in the TaskRunner process.
        # Point imports at the frozen launcher, rather than mutable source files.
        overrides.extend([
            '+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=launch.install_cost_tracking',
            f'+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={run}:{ROOT}',
        ])
        if experiment == 'justrl_math':
            overrides.append(f'reward.custom_reward_function.path={run / "reward.py"}')
        with stage("resolve and save training configuration"):
            with initialize_config_dir(config_dir=str(run), version_base=None):
                config = compose(config_name="recipe", overrides=overrides)
            (run / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
        command = [str(PYTHON), "-u", str(run / "train.py"), "--trainer", "--run-id", run_id, *overrides]
        process = subprocess.Popen(command, cwd=run)
        log(f"PPO driver pid={process.pid}; upstream output follows")
        threading.Thread(target=watch_for_engine_failure, args=(run / 'train.log', process), daemon=True).start()
        try:
            code = process.wait()
            log(f"PPO driver exited with code={code}")
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


