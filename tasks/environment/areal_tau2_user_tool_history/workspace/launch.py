"""Launch the two local model services and run the staged telecom experiment."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    workspace = Path(__file__).resolve().parent
    output = Path(os.environ['LOG_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f'{workspace}/areal:{workspace}/tau2/src',
               TAU2_DATA_DIR=f'{workspace}/tau2/data', WANDB_MODE='offline')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    shared = len(visible) == 1
    processes = []
    logs = []
    started = time.monotonic()
    config = json.loads((workspace / 'configs/rollouts.json').read_text())
    try:
        for index, (role, model, port) in enumerate([
            ('agent', 'Qwen3-4B', 18841), ('user', 'Qwen3-8B', 18842),
        ]):
            log = (output / f'serve-{role}.log').open('w')
            logs.append(log)
            argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', str(workspace / 'models' / model),
                    '--served-model-name', role, '--host', '127.0.0.1', '--port', str(port),
                    '--dtype', 'bfloat16', '--max-model-len', '32768',
                    '--gpu-memory-utilization', '0.42' if shared else '0.85', '--max-num-seqs', '8',
                    '--enable-auto-tool-choice', '--tool-call-parser', 'hermes',
                    '--reasoning-parser', 'qwen3', '--no-enable-log-requests']
            processes.append(subprocess.Popen(argv, env=dict(env, CUDA_VISIBLE_DEVICES=visible[0 if shared else index]),
                                              stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
            while True:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                if processes[-1].poll() is not None:
                    raise RuntimeError(f'The {role} inference service failed during startup')
                if time.monotonic() - started > 900:
                    raise TimeoutError('Model startup exceeded 15 minutes')
                time.sleep(5)
        while True:
            ready = []
            for port in (18841, 18842):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        ready.append(response.status == 200)
                except Exception:
                    ready.append(False)
            if all(ready):
                break
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('An inference service failed during startup')
            if time.monotonic() - started > 900:
                raise TimeoutError('Model startup exceeded 15 minutes')
            time.sleep(5)
        with (output / 'console.log').open('w') as log:
            result = subprocess.run([
                sys.executable, str(workspace / 'run_rollouts.py'),
                '--source', str(workspace / 'areal/examples/tau2/agent.py'),
                '--config', str(workspace / 'configs/rollouts.json'), '--output', str(output),
            ], env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, 3500 - (time.monotonic() - started)))
        if result.returncode:
            raise RuntimeError(f'Rollout collection exited with {result.returncode}')
        summary = json.loads((output / 'summary.json').read_text())
        expected = len(config['task_ids']) * len(config['seeds'])
        before = summary['user_tool_messages_before'] + summary['user_tool_calls_before']
        after = summary['user_tool_messages_after'] + summary['user_tool_calls_after']
        score = after / before if before else 0.0
        metrics = {'step': expected, 'telecom/user_tool_history_retention': score,
                   'episodes': summary['episodes'], 'valid_episodes': summary['valid_episodes'],
                   'mean_reward': summary['mean_reward']}
        (output / 'experiment_metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    main()
